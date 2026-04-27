---
name: Filesystem
version: 1.0.0
description: Local file operations using shell commands.
tags: [local, fs, system]
enabled_by_default: false
---

# Skill: Filesystem

## Tools

### read_file
description: Read a file from disk.
entrypoint: bash:cat {path}
schema:
  type: object
  properties:
    path:
      type: string
      description: Path to the file to read
  required: [path]

### list_dir
description: List files in a directory.
entrypoint: bash:ls -la {path}
schema:
  type: object
  properties:
    path:
      type: string
      description: Path to the directory to list
      default: "."
  required: []

### write_file
description: Write content to a file.
entrypoint: bash:echo "{content}" > {path}
schema:
  type: object
  properties:
    path:
      type: string
      description: Path to the file to write
    content:
      type: string
      description: Content to write to the file
  required: [path, content]

### create_directory
description: Create a directory (and any missing parents).
entrypoint: bash:mkdir -p {path}
schema:
  type: object
  properties:
    path:
      type: string
      description: Path of the directory to create
  required: [path]

### append_file
description: Append content to a file.
entrypoint: bash:echo "{content}" >> {path}
schema:
  type: object
  properties:
    path:
      type: string
      description: Path to the file to append to
    content:
      type: string
      description: Content to append to the file
  required: [path, content]

## Permissions
- local_shell

## Examples
User: "List files in the current directory"
Tool: list_dir()

User: "Show the content of config.yaml"
Tool: read_file(path="config.yaml")
