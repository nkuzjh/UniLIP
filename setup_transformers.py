import os
import shutil
import transformers
import site

def patch_transformers():
    # 1. 获取目标路径 (Conda环境中的transformers目录)
    target_dir = os.path.dirname(transformers.__file__)
    print(f"Target transformers directory: {target_dir}")

    # 2. 获取源路径 (你的项目中的修改代码)
    # 假设脚本在项目根目录，源文件在 ./unilip/openpi_src/...
    current_dir = os.getcwd()
    source_dir = os.path.join(current_dir, "unilip/openpi_src/models_pytorch/transformers_replace")

    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found at {source_dir}")
        return

    # 3. 遍历并复制
    print("Start patching...")
    for root, dirs, files in os.walk(source_dir):
        # 计算相对路径，以便在目标目录保持相同结构
        rel_path = os.path.relpath(root, source_dir)
        target_subdir = os.path.join(target_dir, rel_path)

        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_subdir, file)
            print(f"Copying {file} -> {dst_file}")
            shutil.copy2(src_file, dst_file)

    print("✅ Patching completed successfully!")

if __name__ == "__main__":
    patch_transformers()