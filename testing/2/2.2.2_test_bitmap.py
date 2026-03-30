import numpy as np

# Load the final file
data = np.load('chinese_bitmaps_40x40.npz', allow_pickle=True)

# Check what's inside
print("Keys in file:", list(data.keys()))
print("\nShapes:")
print(f"  bitmaps: {data['bitmaps'].shape}")
print(f"  chars: {data['chars'].shape}")
print(f"  size: {data['size']}")

# Get the data
bitmaps = data['bitmaps']  # Boolean array (total_chars, 40, 40)
chars = data['chars']      # Character array
size = data['size']        # Should be 40

print(f"\n✓ Loaded {len(chars)} characters")
print(f"✓ Bitmap size: {size}x{size}")

# Test a character
zhong_idx = np.where(chars == '鑀')[0][0]
zhong_bitmap = bitmaps[zhong_idx]

print(f"\n'中' uses {np.sum(zhong_bitmap)} pixels out of {size*size}")
print("\nFirst 10 rows (first 20 columns):")
for i in range(min(40, size)):
    row = zhong_bitmap[i][:40]
    row_str = ''.join(['█' if p else '·' for p in row])
    print(f"  {row_str}")

# File size
import os
file_size = os.path.getsize('chinese_bitmaps_40x40.npz') / (1024*1024)
print(f"\n✓ File size: {file_size:.2f} MB")