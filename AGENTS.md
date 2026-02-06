# Collaboration Guidelines

## 1) Bullet keys for clarity
When you provide **Findings**, **Questions**, **Lists**, or any multi-item bullets, prefix **each item** with a **unique key**. (OBS you can use any "keys" as bullets (*, -, ints. etc), not limited to
"
a1)... 
a2)...
"

Example:

Findings  
a1) ...  
a2) ...  

Questions  
b1) ...  
b2) ...  

Rules:
- Keys only need to be unique **within the current message**.
- Keys may be reused in later messages.

## 2) Code gating with flags
I will control whether you write code using these flags:

- **(no code)**: Do not implement or output new code. Only discuss, explain, review, or ask/answer questions.
- **(code ok)**: Code is allowed.

If a message includes **(no code)**, wait until I explicitly send **(code ok)** before you start implementing.

## 3) Plan-first implementation
Process:
- First, provide a **high-level implementation plan** (steps + files/functions to touch).
- Then wait for my approval/adjustments.
- Only after I approve the plan should you write or modify code. 

## 4) Plan approved
if I add PA (plan approved), you can go with code (don't need to wait for code ok here)
PAP (plan approved -> push to github), push the latest update once you are done (with a simple comment)

## 5) Lovable tickets workflow
When we have a new Lovable task, add a new numbered markdown file in `lovable/tickets` (e.g., `1.md`, `2.md`, ...). Do not create templates. When a ticket is solved, admin will add a `_solved` suffix to the filename.

## 6) Git push access in this environment
Push access depends on the SSH key stored in this environment. Any Codex session can push only if it runs in the same environment where the key and SSH config exist. This applies to any repo path opened in this environment (for example `/mnt/c/Users/Dell/Documents/VSC/App/newApp`). Removing the key disables push until a new key is added.

## 7) UDO shortcut for execution
If a message ends with `UDO`, treat it as approval to execute the actions you propose without waiting for a separate confirmation. Always summarize what you did afterward. UDO is if we (for example) debug, and you suggest some CLI commands -> UDO. PA/PAP is for code