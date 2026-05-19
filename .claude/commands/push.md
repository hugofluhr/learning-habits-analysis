Push the current branch to origin.

Steps:
1. Run `git remote get-url origin` to check the remote URL.
2. Run `git status` to confirm the current branch and that there is nothing unexpected staged or uncommitted.
3. Push with `git push origin <current-branch>`.
4. Report the result.

## Remote-dependent behaviour

**HTTPS remote** (e.g. `https://github.com/...`):
You are on a remote VM (e.g. a cloud instance accessed via VS Code Server). Credentials are supplied automatically via the VS Code askpass mechanism — no SSH key or token needed. This is the expected setup on this machine.

**SSH remote** (e.g. `git@github.com:...`):
You are on a local machine where SSH keys are available. `git push` uses the SSH agent directly. Do NOT attempt to configure or use the VS Code askpass mechanism here — it is not needed and may not be present.

**HTTPS remote on a local machine** (no VS Code Server):
Do not use this skill. There is no VS Code credential cache to rely on. Ask the user to push manually or set up `gh auth setup-git` first.
