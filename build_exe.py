"""
Script to build the Sales Forecasting Application as an executable
Run this script: python build_exe.py

The .app will be created in the project root and will use:
- models/ folder (same directory as .app)
- mlruns/ folder (same directory as .app)

To distribute, copy these together:
- Sales_Forecasting_App.app
- models/
- mlruns/
"""
import PyInstaller.__main__
import os
import shutil

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

PyInstaller.__main__.run([
    'app_qt.py',
    '--name=Sales_Forecasting_App',
    '--windowed',  # No console window
    '--onedir',   # Faster startup (folder with dependencies)
    f'--add-data={os.path.join(current_dir, "data")}{os.pathsep}data',
    # Models folder kept external to reduce exe size
    f'--add-data={os.path.join(current_dir, "utils")}{os.pathsep}utils',
    '--exclude-module=PyQt5',  # Exclude PyQt5 to avoid conflicts
    '--exclude-module=PySide2',
    '--exclude-module=PySide6',
    '--hidden-import=sklearn.utils._weight_vector',
    '--hidden-import=joblib',
    '--hidden-import=pandas',
    '--hidden-import=numpy',
    '--hidden-import=matplotlib',
    '--hidden-import=statsmodels',
    '--hidden-import=holidays.countries',
    '--hidden-import=holidays.financial',
    '--collect-all=holidays',
    '--noconfirm',  # Overwrite without asking
])

print("\n" + "="*70)
print("Build complete! Moving .app to project root...")
print("="*70)

# Move .app from dist/ to project root
dist_app = os.path.join(current_dir, 'dist', 'Sales_Forecasting_App.app')
root_app = os.path.join(current_dir, 'Sales_Forecasting_App.app')

if os.path.exists(root_app):
    shutil.rmtree(root_app)

if os.path.exists(dist_app):
    shutil.move(dist_app, root_app)
    print(f"✓ Moved .app to: {root_app}")
    
    # Clean up dist folder (optional)
    dist_folder = os.path.join(current_dir, 'dist', 'Sales_Forecasting_App')
    if os.path.exists(dist_folder):
        shutil.rmtree(dist_folder)
        print(f"✓ Cleaned up dist folder")

print("\n" + "="*70)
print("READY TO USE!")
print("="*70)
print(f"App location: Sales_Forecasting_App.app")
print(f"Models folder: models/")
print(f"MLflow folder: mlruns/")
print("\nThe app will automatically use models/ and mlruns/ from the same folder.")
print("="*70)
