# Security

## Reporting a vulnerability

Please report security problems privately through GitHub's "Report a vulnerability" button
(Security tab of the repository) rather than in a public issue. You should get an answer within
a week.

## Things to know when using the simulator

- **The browser UI** (`sds ui`) listens on this computer only (127.0.0.1) by default. Anyone who
  can open it can run the examples with any parameters and fill the disk with results.
  Example 12 also lets them make Python import a module by name. Expose it with `--address` only
  on a network you trust.
- **Model roles** (`simulator.ai`) send prompts to whatever model function you supply and log
  prompts and outputs in plain text. Keep API keys in your environment, not in configs or
  logs.
- **The model checker** runs model code in a subprocess with a reduced environment. That limits
  accidents, but it is not a sandbox: check untrusted model code yourself before running it.
- **Dependencies:** the package sets security floors where a known fix matters, such as
  `pillow>=13.0`. Install into a dedicated environment and keep it updated with the update
  scripts. Dependabot watches the dependencies and the CI actions.
