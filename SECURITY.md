# Security

Do not commit `.env`, API keys, local logs, browser profiles, or raw experiment dumps.

If a key was ever committed, revoke or rotate it immediately. Removing the file in a later commit does not remove it from Git history.

Before publishing, run a secret scan over the repository and inspect generated CSV/log outputs for local paths or sensitive prompt content.

