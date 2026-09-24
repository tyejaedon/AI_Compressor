# Random Search Profile

`src/training/random_search_hyperparams.py` supports a centralized JSON profile that controls:

- sampled hyperparameter spaces (`sample_space`)
- fixed trainer arguments by modality/profile (`fixed_resource_args`)
- summary range keys (`summary_keys`)

Default profile path:

- `documentation/random_search_profile.json`

## Usage

```zsh
python3 src/training/random_search_hyperparams.py --modality image --image-trainer standard --resource-profile tiny --trials 10
python3 src/training/random_search_hyperparams.py --modality audio --resource-profile balanced --trials 12
```

Use a custom profile file:

```zsh
python3 src/training/random_search_hyperparams.py --modality video --search-config documentation/random_search_profile.json --trials 8
```

## Profile Schema (high level)

- `sample_space`
  - `image.standard`, `image.lossy`, `audio`, `video`
  - each key maps to a literal value or sampling spec:
    - list: random choice
    - `{ "choices": [...] }`
    - `{ "log_uniform": [low, high], "round": n }`
    - `{ "uniform": [low, high], "round": n }`
    - `{ "int": [low, high] }`
    - `{ "int_step": [low, high, step] }`
- `fixed_resource_args`
  - modality/profile (and image trainer) specific CLI args appended to trainer commands
- `summary_keys`
  - keys included in top-trial range summaries

## Notes

- If `--search-config` points to a missing file, the script falls back to built-in defaults.
- `params_to_cli_args` auto-converts sampled keys to CLI flags (`model_base_filters` -> `--model-base-filters`).

