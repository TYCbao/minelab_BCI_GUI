from PyInstaller.utils.hooks import collect_data_files, copy_metadata, collect_submodules

# Collect all submodules and data files for mne
datas = collect_data_files('mne')
hiddenimports = collect_submodules('mne')
datas += copy_metadata('mne')

# 添加 lazy_loader 作為隱藏導入
hiddenimports += ['lazy_loader']
