Always make the smallest possible change to satisfy the request.
Keep the code DRY and follow best-practices.
This code is built to last a long time; prioritize longevity and regularity over experiments and hacks.
Keep in mind YAGNI; don't build anything you don't need.

<!-- Codex-reliability:binary-instructions managed section - DO NOT EDIT -->
## Codex-reliability Binary

The `Codex-reliability` binary for this project is located at:

    .Codex-reliability/bin/Codex-reliability

Always use this path when running commands. Do NOT use bare `Codex-reliability`,
do NOT use paths containing `~/.Codex-reliability/`, and do NOT use `$PLUGIN_ROOT_DIR`
or any other variable to construct the path.

Example usage:

    .Codex-reliability/bin/Codex-reliability work list
    .Codex-reliability/bin/Codex-reliability work next
    .Codex-reliability/bin/Codex-reliability work on <id>
    .Codex-reliability/bin/Codex-reliability work update <id> --status complete
<!-- end Codex-reliability:binary-instructions -->
