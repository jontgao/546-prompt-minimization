from huggingface_hub import HfApi

if __name__ == '__main__':
    api = HfApi()

    api.upload_large_folder(
        folder_path='runs/',
        repo_id="SuperComputer/PromptMinimization",
        repo_type="model",
        ignore_patterns=['**/*.logs', '**/checkpoint_*'],  # Ignore all text logs
    )