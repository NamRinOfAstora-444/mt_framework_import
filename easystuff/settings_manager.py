"""
settings_manager.py

Port note for EasyStuff/SettingsManager.ms.

The original SettingsManager persisted 3ds Max rollout values (checkbox states,
spinner values, picked nodes) into a single .ini file so the UI remembered its
state between sessions.

Blender does not need this: operator properties (the import-dialog options on
IMPORT_OT_mt_framework_mod) are persisted automatically by Blender between runs,
and tool settings live in the .blend / preferences. So there is nothing to port
here -- this file exists only to document that decision.

If per-project presets are ever wanted, use an AddonPreferences subclass or
Blender's built-in operator preset system rather than re-implementing an .ini
manager.
"""
