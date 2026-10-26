"""
lmt_import.py
Applies decoded LMT animation (lmt_format.py) onto a Blender armature as Actions.

Loop handling:
    LoopFrame == 0
        -> toàn bộ animation đã là loop
        -> chỉ tạo anim_XXX_loop

    0 < LoopFrame < FrameCount
        -> tạo:
           anim_XXX       = phần trước loop
           anim_XXX_loop  = phần loop

    LoopFrame không hợp lệ
        -> animation bình thường
"""

import bpy
import mathutils

from .lmt_format import LMTHeader


# ============================================================
# SETTINGS
# ============================================================

_ANIM_QUAT_CONJUGATE = False


# MT Framework -> Blender coordinate conversion.
MTFCORMAT_4 = mathutils.Matrix((
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))


# ============================================================
# REST / COORDINATE HELPERS
# ============================================================

def _bone_rest_relative(bone):
    """Return the bone rest transform in its parent's local space."""

    if bone.parent is not None:
        return bone.parent.matrix_local.inverted() @ bone.matrix_local

    return bone.matrix_local.copy()


def _root_target_position(v):
    """Convert MT Framework root/world position into Blender space."""
    return MTFCORMAT_4 @ v


# ============================================================
# BONE 255
# ============================================================

def _root255_bone(arm_obj):
    """
    Find the synthetic MT root/reference bone.

    Bone 255 is treated as a reference axis only.
    """

    for bone in arm_obj.data.bones:
        try:
            if int(bone.get("mtf_index", -1)) == 255:
                return bone
        except (TypeError, ValueError):
            pass

    root_name = arm_obj.get("mtf_root_bone", "bone255")
    return arm_obj.data.bones.get(root_name)


def _root255_basis(arm_obj):
    """
    Return bone255's rest axes.

    bone255 is a non-parenting reference bone.
    Its orientation determines the direction in which MT root motion
    is placed in Blender.
    """

    root_bone = _root255_bone(arm_obj)

    if root_bone is None:
        return MTFCORMAT_4.to_3x3()

    return root_bone.matrix_local.to_3x3().copy()


def _root255_target_position(arm_obj, v, root_basis=None):
    """
    Convert an MT root-motion vector using bone255's rest axes.

    This does NOT use Blender world origin as the reference direction.
    """

    basis = (
        root_basis
        if root_basis is not None
        else _root255_basis(arm_obj)
    )

    return basis @ v


def _apply_root_motion_to_object(
    arm_obj,
    v,
    root_basis,
    frame,
):
    """
    Apply BoneID 255 / Usage 4 root motion to the Armature Object.

    bone255 itself is not animated and is not the parent of the skeleton.
    It only defines the reference axes.

    The converted motion is keyed on armature.location.
    """

    target = _root255_target_position(
        arm_obj,
        v,
        root_basis,
    )

    # Preserve the object's location before animation import.
    base = arm_obj.get(
        "mtf_root_motion_base_location"
    )

    if base is None:
        base = tuple(
            float(x)
            for x in arm_obj.location
        )

        arm_obj["mtf_root_motion_base_location"] = base

    base_vec = mathutils.Vector(base)

    arm_obj.location = base_vec + target

    arm_obj.keyframe_insert(
        "location",
        frame=frame,
        group="Root Motion",
    )


# ============================================================
# ROTATION HELPERS
# ============================================================

def _rotation_from_mt_quaternion(q):
    """
    Convert decoded MT quaternion into a Blender rotation matrix.
    """

    return q.to_matrix().to_4x4()


def _root_controller_rotation(q):
    """
    Equivalent of the original MaxScript root rotation conversion:

        inverse((b.value as matrix3) * MTFCormat).rotation
    """

    bmat = _rotation_from_mt_quaternion(q)

    out = (
        bmat.inverted()
        @ MTFCORMAT_4.inverted()
    )

    return out.to_quaternion().normalized()


def _local_controller_rotation(q):
    """
    Child-bone rotations are already parent-local.
    """

    return q.normalized()


# ============================================================
# LOCATION HELPERS
# ============================================================

def _pose_location_from_local_target(
    pbone,
    target_local,
):
    """
    Convert an absolute parent-local position into Blender pose location.
    """

    rest = _bone_rest_relative(
        pbone.bone
    )

    rest_rot = rest.to_quaternion()

    return (
        rest_rot.inverted()
        @ (
            target_local
            - rest.translation
        )
    )


def _pose_location_from_root_target(
    pbone,
    target_armature_space,
):
    """
    Convert root/world target into Blender root pose location.
    """

    rest_world = pbone.bone.matrix_local

    rest_rot = rest_world.to_quaternion()

    return (
        rest_rot.inverted()
        @ (
            target_armature_space
            - rest_world.translation
        )
    )


# ============================================================
# BONE LOOKUP
# ============================================================

def _find_pose_bone_by_mtf_index(
    arm_obj,
    mtf_index,
):
    """
    Find a Blender pose bone by its skeleton-local MT index.

    Primary:
        Bone["mtf_index"]

    Fallback:
        deterministic armature bone order for older .blend files.
    """

    try:
        target = int(mtf_index)

    except (TypeError, ValueError):
        return None

    # --------------------------------------------------------
    # Preferred persistent mapping
    # --------------------------------------------------------

    for pbone in arm_obj.pose.bones:

        try:
            if int(
                pbone.bone.get(
                    "mtf_index",
                    -1,
                )
            ) == target:

                return pbone

        except (
            TypeError,
            ValueError,
        ):
            continue

    bones = list(
        arm_obj.data.bones
    )

    if not bones:
        return None

    # --------------------------------------------------------
    # Bone255
    # --------------------------------------------------------

    if target == 255:

        for bone in bones:

            if (
                bone.parent is None
                and not bone.use_deform
            ):
                return arm_obj.pose.bones.get(
                    bone.name
                )

        return arm_obj.pose.bones.get(
            bones[0].name
        )

    # --------------------------------------------------------
    # Compatibility fallback
    # --------------------------------------------------------

    ordered = []

    for bone in bones:

        if (
            bone.parent is None
            and not bone.use_deform
        ):
            continue

        ordered.append(bone)

    if 0 <= target < len(ordered):

        return arm_obj.pose.bones.get(
            ordered[target].name
        )

    return None


def _resolve_pose_bone(
    arm_obj,
    remaptable,
    rootid,
    bone_id,
):
    """
    Resolve an LMT bone ID into a Blender pose bone.

    Bone255 is a synthetic root-motion controller.
    """

    if bone_id is None:
        return None

    # --------------------------------------------------------
    # Root / synthetic root
    # --------------------------------------------------------

    if (
        bone_id == rootid
        or (
            rootid == 255
            and bone_id == 255
        )
    ):

        game_root = arm_obj.get(
            "mtf_game_root_bone"
        )

        if game_root:

            pbone = arm_obj.pose.bones.get(
                game_root
            )

            if pbone is not None:
                return pbone

        # Compatibility fallback
        root_name = arm_obj.get(
            "mtf_root_bone",
            "bone255",
        )

        if root_name != "bone255":

            pbone = arm_obj.pose.bones.get(
                root_name
            )

            if pbone is not None:
                return pbone

        # Last resort:
        # first deforming root
        for pbone in arm_obj.pose.bones:

            if (
                pbone.bone.parent is None
                and pbone.bone.use_deform
            ):
                return pbone

        return None

    # --------------------------------------------------------
    # Remap table
    # --------------------------------------------------------

    if (
        remaptable
        and 0 <= bone_id < len(remaptable)
    ):

        scene_idx = int(
            remaptable[bone_id]
        )

        if (
            scene_idx == 255
            and bone_id != rootid
        ):
            return None

        return _find_pose_bone_by_mtf_index(
            arm_obj,
            scene_idx,
        )

    # --------------------------------------------------------
    # Direct fallback
    # --------------------------------------------------------

    return _find_pose_bone_by_mtf_index(
        arm_obj,
        bone_id,
    )


# ============================================================
# ANIMATION APPLICATION
# ============================================================

def apply_animation(
    arm_obj,
    ablock,
    action,
    start_frame=1,
    conjugate=False,
    include_location=True,
    key_time_start=None,
    key_time_end=None,
    key_time_offset=0,
):
    """
    Apply one decoded LMT animation block.

    key_time_start / key_time_end:
        source LMT frame range to import.

    key_time_offset:
        offset added after start_frame.

    These are used to split partially looping animations.
    """

    remaptable = list(
        arm_obj.get(
            "mtf_remaptable",
            [],
        )
    )

    rootid = int(
        arm_obj.get(
            "mtf_lmt_rootid",
            255,
        )
    )

    root255_basis = _root255_basis(
        arm_obj
    )

    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()

    arm_obj.animation_data.action = action

    # --------------------------------------------------------
    # Key filtering
    # --------------------------------------------------------

    def _key_in_range(key):

        if (
            key_time_start is not None
            and key.time < key_time_start
        ):
            return False

        if (
            key_time_end is not None
            and key.time > key_time_end
        ):
            return False

        return True

    def _dst_frame(key):

        return (
            key.time
            + start_frame
            + key_time_offset
        )

    # --------------------------------------------------------
    # Tracks
    # --------------------------------------------------------

    for track in ablock.KeyBuffersRef:

        usage = int(track.Usage)

        is_root_track = (
            track.BoneID == rootid
        )

        # ====================================================
        # ROOT MOTION
        # ====================================================

        if (
            is_root_track
            and usage == 4
            and include_location
        ):

            for key in track.FrameArray:

                if not _key_in_range(key):
                    continue

                v = key.value

                if not isinstance(
                    v,
                    mathutils.Vector,
                ):
                    continue

                _apply_root_motion_to_object(
                    arm_obj,
                    v,
                    root255_basis,
                    _dst_frame(key),
                )

            continue

        # ====================================================
        # NORMAL BONE
        # ====================================================

        pbone = _resolve_pose_bone(
            arm_obj,
            remaptable,
            rootid,
            track.BoneID,
        )

        if (
            pbone is None
            or not track.FrameArray
        ):
            continue

        is_game_root = (
            bool(
                pbone.bone.get(
                    "mtf_game_root",
                    False,
                )
            )
            or
            pbone.name
            == arm_obj.get(
                "mtf_game_root_bone"
            )
        )

        # ====================================================
        # ROTATION
        # ====================================================

        if usage in (0, 3):

            pbone.rotation_mode = (
                "QUATERNION"
            )

            for key in track.FrameArray:

                if not _key_in_range(key):
                    continue

                q = key.value

                if not isinstance(
                    q,
                    mathutils.Quaternion,
                ):
                    continue

                q = q.normalized()

                if conjugate:
                    q = q.conjugated()

                if (
                    usage == 3
                    or is_game_root
                    or is_root_track
                ):

                    basis = (
                        _root_controller_rotation(q)
                    )

                else:

                    basis = (
                        _local_controller_rotation(q)
                    )

                pbone.rotation_quaternion = basis

                pbone.keyframe_insert(
                    "rotation_quaternion",
                    frame=_dst_frame(key),
                    group=pbone.name,
                )

        # ====================================================
        # LOCATION
        # ====================================================

        elif (
            usage in (1, 4)
            and include_location
        ):

            for key in track.FrameArray:

                if not _key_in_range(key):
                    continue

                v = key.value

                if not isinstance(
                    v,
                    mathutils.Vector,
                ):
                    continue

                if (
                    usage == 1
                    and is_game_root
                ):

                    target = (
                        _root255_target_position(
                            arm_obj,
                            v,
                            root255_basis,
                        )
                    )

                    loc = (
                        _pose_location_from_root_target(
                            pbone,
                            target,
                        )
                    )

                else:

                    loc = (
                        _pose_location_from_local_target(
                            pbone,
                            v,
                        )
                    )

                pbone.location = loc

                pbone.keyframe_insert(
                    "location",
                    frame=_dst_frame(key),
                    group=pbone.name,
                )

        # ====================================================
        # SCALE
        # ====================================================

        elif usage == 2:

            for key in track.FrameArray:

                if not _key_in_range(key):
                    continue

                v = key.value

                if not isinstance(
                    v,
                    mathutils.Vector,
                ):
                    continue

                pbone.scale = v

                pbone.keyframe_insert(
                    "scale",
                    frame=_dst_frame(key),
                    group=pbone.name,
                )


# ============================================================
# ACTION HELPERS
# ============================================================

def _configure_action_range(
    action,
    start_frame,
    frame_count,
):
    """
    Configure Blender Action frame range.
    """

    end_frame = (
        start_frame
        + max(
            int(frame_count) - 1,
            0,
        )
    )

    if hasattr(
        action,
        "use_frame_range",
    ):

        try:

            action.use_frame_range = True

            action.frame_start = float(
                start_frame
            )

            action.frame_end = float(
                max(
                    end_frame,
                    start_frame,
                )
            )

        except Exception:
            pass

    return end_frame


def _set_action_metadata(
    action,
    animation_index,
    frame_count,
    loop_frame,
    is_loop,
    full_loop=False,
    source_action=None,
):
    """
    Store MT/LMT metadata on Action.
    """

    action["mtf_animation_index"] = int(
        animation_index
    )

    action["mtf_frame_count"] = int(
        frame_count
    )

    action["mtf_loop_frame"] = int(
        loop_frame
    )

    action["mtf_is_loop"] = bool(
        is_loop
    )

    action["mtf_full_loop"] = bool(
        full_loop
    )

    if source_action is not None:

        action["mtf_source_action"] = (
            source_action.name
        )


# ============================================================
# LMT IMPORT
# ============================================================

def import_lmt(
    filepath,
    arm_obj,
    context,
    x64=False,
    single_id=None,
    name_prefix="anim",
    conjugate=False,
    include_location=True,
    set_timeline=True,
):
    """
    Import all animations from an LMT file.

    Loop rules:

        LoopFrame == 0
            -> full animation is already a loop
            -> ONLY:
               anim_XXX_loop

        0 < LoopFrame < FrameCount
            -> split:
               anim_XXX
               anim_XXX_loop

        invalid LoopFrame
            -> normal:
               anim_XXX
    """

    start_frame = 1

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    with open(
        filepath,
        "rb",
    ) as fh:

        data = fh.read()

    from .easystuff.read_stream import BinaryReader

    reader = BinaryReader(data)

    header = LMTHeader(
        reader,
        x64=x64,
    )

    # --------------------------------------------------------
    # Animation IDs
    # --------------------------------------------------------

    indices = (
        [single_id]
        if single_id is not None
        else range(
            len(
                header.Offsettable
            )
        )
    )

    count = 0
    active_end = None

    # Numeric ordering:
    # anim_000
    # anim_001
    # ...
    pad = max(
        len(
            str(
                max(
                    len(header.Offsettable) - 1,
                    0,
                )
            )
        ),
        1,
    )

    # ========================================================
    # IMPORT EACH ANIMATION
    # ========================================================

    for i in indices:

        if (
            i < 0
            or i >= len(
                header.Offsettable
            )
        ):
            continue

        # ----------------------------------------------------
        # Decode block
        # ----------------------------------------------------

        try:

            ablock = (
                header.get_animation_from_id(i)
            )

        except Exception as exc:

            print(
                "LMT block %d failed: %s"
                % (
                    i,
                    exc,
                )
            )

            continue

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        base_name = (
            "%s_%0*d"
            % (
                name_prefix,
                pad,
                i,
            )
        )

        frame_count = int(
            getattr(
                ablock,
                "FrameCount",
                0,
            )
            or 0
        )

        raw_loop_frame = getattr(
            ablock,
            "LoopFrame",
            None,
        )

        if raw_loop_frame is None:

            loop_frame = -1

        else:

            try:
                loop_frame = int(
                    raw_loop_frame
                )
            except (
                TypeError,
                ValueError,
            ):
                loop_frame = -1

        # ----------------------------------------------------
        # Detect loop type
        # ----------------------------------------------------

        full_loop = (
            frame_count > 0
            and loop_frame == 0
        )

        partial_loop = (
            frame_count > 0
            and loop_frame > 0
            and loop_frame < frame_count
        )

        # ====================================================
        # FULL LOOP
        # ====================================================

        if full_loop:

            try:

                action = bpy.data.actions.new(
                    base_name + "_loop"
                )

                action.use_fake_user = True

                apply_animation(
                    arm_obj,
                    ablock,
                    action,
                    start_frame=start_frame,
                    conjugate=conjugate,
                    include_location=include_location,
                    key_time_start=0,
                    key_time_end=frame_count - 1,
                    key_time_offset=0,
                )

                _set_action_metadata(
                    action,
                    i,
                    frame_count,
                    0,
                    True,
                    full_loop=True,
                )

                active_end = (
                    _configure_action_range(
                        action,
                        start_frame,
                        frame_count,
                    )
                )

                count += 1

                print(
                    "LMT full loop: %s [0..%d]"
                    % (
                        action.name,
                        frame_count - 1,
                    )
                )

            except Exception as exc:

                print(
                    "Applying LMT block %d failed: %s"
                    % (
                        i,
                        exc,
                    )
                )

                import traceback
                traceback.print_exc()

            continue

        # ====================================================
        # PARTIAL LOOP
        # ====================================================

        if partial_loop:

            try:

                # ------------------------------------------------
                # PRE-LOOP
                # ------------------------------------------------

                pre_action = (
                    bpy.data.actions.new(
                        base_name
                    )
                )

                pre_action.use_fake_user = True

                apply_animation(
                    arm_obj,
                    ablock,
                    pre_action,
                    start_frame=start_frame,
                    conjugate=conjugate,
                    include_location=include_location,
                    key_time_start=0,
                    key_time_end=loop_frame - 1,
                    key_time_offset=0,
                )

                pre_count = loop_frame

                _set_action_metadata(
                    pre_action,
                    i,
                    pre_count,
                    loop_frame,
                    False,
                    full_loop=False,
                )

                pre_action[
                    "mtf_source_frame_count"
                ] = frame_count

                _configure_action_range(
                    pre_action,
                    start_frame,
                    pre_count,
                )

                # ------------------------------------------------
                # LOOP
                # ------------------------------------------------

                loop_action = (
                    bpy.data.actions.new(
                        base_name
                        + "_loop"
                    )
                )

                loop_action.use_fake_user = True

                apply_animation(
                    arm_obj,
                    ablock,
                    loop_action,
                    start_frame=start_frame,
                    conjugate=conjugate,
                    include_location=include_location,
                    key_time_start=loop_frame,
                    key_time_end=frame_count - 1,
                    key_time_offset=-loop_frame,
                )

                loop_count = (
                    frame_count
                    - loop_frame
                )

                _set_action_metadata(
                    loop_action,
                    i,
                    loop_count,
                    0,
                    True,
                    full_loop=False,
                    source_action=pre_action,
                )

                loop_action[
                    "mtf_source_loop_frame"
                ] = loop_frame

                loop_action[
                    "mtf_source_frame_count"
                ] = frame_count

                _configure_action_range(
                    loop_action,
                    start_frame,
                    loop_count,
                )

                # The pre-loop Action is active.
                if (
                    arm_obj.animation_data
                    is None
                ):

                    arm_obj.animation_data_create()

                arm_obj.animation_data.action = (
                    pre_action
                )

                active_end = (
                    start_frame
                    + max(
                        pre_count - 1,
                        0,
                    )
                )

                count += 1

                print(
                    "LMT loop split: "
                    "%s [0..%d] | "
                    "%s [%d..%d]"
                    % (
                        pre_action.name,
                        loop_frame - 1,
                        loop_action.name,
                        loop_frame,
                        frame_count - 1,
                    )
                )

            except Exception as exc:

                print(
                    "Applying LMT block %d failed: %s"
                    % (
                        i,
                        exc,
                    )
                )

                import traceback
                traceback.print_exc()

            continue

        # ====================================================
        # NORMAL ANIMATION
        # ====================================================

        try:

            action = bpy.data.actions.new(
                base_name
            )

            action.use_fake_user = True

            # If FrameCount is known, explicitly import only
            # its valid frame range.
            if frame_count > 0:

                key_time_start = 0
                key_time_end = (
                    frame_count - 1
                )

            else:

                key_time_start = None
                key_time_end = None

            apply_animation(
                arm_obj,
                ablock,
                action,
                start_frame=start_frame,
                conjugate=conjugate,
                include_location=include_location,
                key_time_start=key_time_start,
                key_time_end=key_time_end,
                key_time_offset=0,
            )

            _set_action_metadata(
                action,
                i,
                frame_count,
                loop_frame,
                False,
                full_loop=False,
            )

            active_end = (
                _configure_action_range(
                    action,
                    start_frame,
                    frame_count,
                )
            )

            count += 1

        except Exception as exc:

            print(
                "Applying LMT block %d failed: %s"
                % (
                    i,
                    exc,
                )
            )

            import traceback
            traceback.print_exc()

    # ========================================================
    # SCENE TIMELINE
    # ========================================================

    if (
        set_timeline
        and count
        and active_end is not None
        and context is not None
    ):

        scene = getattr(
            context,
            "scene",
            None,
        )

        if scene is not None:

            try:

                scene.frame_start = (
                    start_frame
                )

                scene.frame_end = max(
                    active_end,
                    start_frame,
                )

                scene.frame_current = (
                    start_frame
                )

            except Exception:
                pass

    return count