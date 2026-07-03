# Remotes

A remote is a named reference to another copy of the repository, typically on a
server such as GitHub. The default remote created by `git clone` is named
`origin`.

## Fetch, pull, and push

- `git fetch` downloads new commits and refs from a remote but does not change
  your working tree or current branch.
- `git pull` runs `git fetch` and then integrates the changes into your current
  branch, by merging by default (or rebasing with `git pull --rebase`).
- `git push` uploads your local commits to the remote branch.

## Tracking branches

A local branch can track a remote branch, which lets `git pull` and `git push`
work without extra arguments. Set the upstream with
`git push -u origin <branch>` the first time you push, or with
`git branch --set-upstream-to=origin/<branch>`.

## Inspecting remotes

`git remote -v` lists the configured remotes and their URLs. `git remote add
<name> <url>` adds a new remote.
