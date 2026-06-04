---
trigger: always_on
---

## workspace-scope

Access and modify only files within the active workspace.

## forbidden-operations

Without explicit user permission, the agent must not:

1. Access paths outside the workspace.
2. Modify, copy, or delete files outside the workspace.
3. Scan or infer system directory structures.

## file-operation-disclosure

All file operations must:

1. Explicitly state the target files.
2. Request confirmation when appropriate.
