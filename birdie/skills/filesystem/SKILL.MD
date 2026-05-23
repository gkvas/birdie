---
name: Filesystem
version: 1.0.0
description: Read, write, and manage local files and directories.
tags: [local, fs, system]
enabled_by_default: false
---

## Tools

### read_file
description: Read the full contents of a file from disk. Best for files under ~200 lines. For larger files prefer the Shell skill with `sed -n 'START,ENDp' path` to read specific ranges or `grep -n` to search.
entrypoint: bash:cat {path}
schema:
  type: object
  properties:
    path:
      type: string
      description: Path to the file to read
  required: [path]

### list_dir
description: List files and directories at the given path in long format (permissions, size, modification time). Defaults to the current directory when path is omitted or ".".
entrypoint: bash:ls -la {path}
schema:
  type: object
  properties:
    path:
      type: string
      description: Path to the directory to list (default ".")
      default: "."
  required: []

### write_file
description: Write content to a file, replacing any existing content. Warning - uses `echo … >` internally so content containing quotes, backslashes, or newlines will be corrupted. For complex content use the Shell skill with a heredoc instead.
entrypoint: bash:echo "{content}" > {path}
schema:
  type: object
  properties:
    path:
      type: string
      description: Path to the file to write (created if it does not exist)
    content:
      type: string
      description: Content to write; avoid quotes, backslashes, and newlines
  required: [path, content]

### append_file
description: Append content to the end of a file. Same quoting limitations as write_file - content with quotes, backslashes, or newlines will be corrupted. Use the Shell skill with a heredoc for complex content.
entrypoint: bash:echo "{content}" >> {path}
schema:
  type: object
  properties:
    path:
      type: string
      description: Path to the file to append to (created if it does not exist)
    content:
      type: string
      description: Content to append; avoid quotes, backslashes, and newlines
  required: [path, content]

### create_directory
description: Create a directory and any missing parent directories (equivalent to `mkdir -p`). Safe to call even if the directory already exists.
entrypoint: bash:mkdir -p {path}
schema:
  type: object
  properties:
    path:
      type: string
      description: Path of the directory to create
  required: [path]
