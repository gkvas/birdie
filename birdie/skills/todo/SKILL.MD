---
name: ToDo
version: 1.0.0
description: Step-by-step planning and progress tracking for every user request.
tags: [planning, meta]
enabled_by_default: false
always_inject: true
---

For **every** user request, follow this protocol without exception:

1. **Plan first** — Call `todo_create_plan` with a concrete, ordered list of steps you will take.
2. **Execute step by step** — Carry out each step using whatever tools are available.
3. **Track progress** — After completing each step, call `todo_complete_step` with the step number and a one-line summary of what was done or found.
4. **Respond** — Only after all steps are marked complete, give your final answer.

Keep steps short and action-oriented (e.g. "Read config file", "List directory", "Run ssh command"). Do not skip planning even for simple requests.

## Tools

### todo_create_plan
description: Create and display the step-by-step plan for the current request. Call this before taking any other action. `steps` is an ordered list of short, action-oriented strings describing each step you will execute.
entrypoint: python:birdie.skills.todo.tools.create_plan
schema:
  type: object
  properties:
    steps:
      type: array
      items:
        type: string
      description: Ordered list of steps to execute
  required: [steps]

### todo_complete_step
description: Mark a plan step as complete and record what was done. Call this immediately after finishing each step. `step_number` is 1-based and must match a step from the active plan.
entrypoint: python:birdie.skills.todo.tools.complete_step
schema:
  type: object
  properties:
    step_number:
      type: integer
      description: The 1-based index of the completed step
    summary:
      type: string
      description: One-line summary of what was done or found
  required: [step_number, summary]
