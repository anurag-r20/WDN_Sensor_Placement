import os
import shutil
from pathlib import Path

def migrate_logs_to_city_folders(logs_dir='logs'):
    """
    Migrate existing log files to city-specific subdirectories.
    Files are named as {city}_{solver}.{extension}
    """
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        print(f"❌ Directory {logs_dir} does not exist")
        return
    
    # Get all files in logs directory
    files = [f for f in logs_path.iterdir() if f.is_file()]
    
    # Group files by city
    cities = set()
    for file in files:
        # Extract city name (first part before underscore)
        parts = file.name.split('_')
        if len(parts) >= 2:
            city = parts[0]
            cities.add(city)
    
    print(f"Found cities: {', '.join(sorted(cities))}\n")
    
    # Create city directories and move files
    moved_count = 0
    for city in sorted(cities):
        city_dir = logs_path / city
        city_dir.mkdir(exist_ok=True)
        print(f"📁 Created/verified directory: {city_dir}")
        
        # Move files for this city
        city_files = [f for f in files if f.name.startswith(f"{city}_")]
        for file in city_files:
            # Remove city prefix from filename
            new_name = file.name.replace(f"{city}_", "")
            dest_path = city_dir / new_name
            
            # Move file
            shutil.move(str(file), str(dest_path))
            moved_count += 1
            print(f"  ✓ Moved: {file.name} → {city}/{new_name}")
        
        print(f"  Total files moved for {city}: {len(city_files)}\n")
    
    print(f"✅ Migration complete! Moved {moved_count} files to city-specific folders.")
    
    # Show final structure
    print("\n📂 New directory structure:")
    for city in sorted(cities):
        city_dir = logs_path / city
        city_files = list(city_dir.iterdir())
        print(f"  {city}/ ({len(city_files)} files)")
        for f in sorted(city_files)[:3]:  # Show first 3 files
            print(f"    - {f.name}")
        if len(city_files) > 3:
            print(f"    ... and {len(city_files) - 3} more")

if __name__ == "__main__":
    migrate_logs_to_city_folders()
