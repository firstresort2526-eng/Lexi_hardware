import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time

class TraditionalChineseBitmapToNPZ:
    def __init__(self, font_path=None, size=40, output_file='chinese_bitmaps.npz'):
        self.size = size
        self.output_file = output_file
        self.font_path = font_path or self._find_traditional_chinese_font()
        
        if not self.font_path:
            print("⚠ No Traditional Chinese font found!")
            print("Installing...")
            os.system('sudo apt update')
            os.system('sudo apt install -y fonts-arphic-ukai fonts-arphic-uming')
            self.font_path = self._find_traditional_chinese_font()
        
        if self.font_path:
            print(f"✓ Using font: {self.font_path}")
        else:
            raise Exception("No Traditional Chinese font found.")
        
        self.start_code = 0x4E00
        self.end_code = 0x9FFF
        self.total_chars = self.end_code - self.start_code + 1
    
    def _find_traditional_chinese_font(self):
        """Find Traditional Chinese fonts"""
        font_paths = [
            '/usr/share/fonts/truetype/arphic/ukai.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                return path
        return None
    
    def render_character(self, char):
        """Render character to 40x40 bitmap - returns boolean array"""
        try:
            img = Image.new('L', (self.size, self.size), 255)
            draw = ImageDraw.Draw(img)
            
            # Auto-adjust font size to fill the space
            font_size = self.size
            
            for attempt in range(3):
                try:
                    font = ImageFont.truetype(self.font_path, font_size)
                    bbox = draw.textbbox((0, 0), char, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    if text_width < self.size * 0.7:
                        font_size = int(font_size * 1.2)
                        continue
                    
                    if text_width > self.size * 0.95 or text_height > self.size * 0.95:
                        font_size = int(font_size * 0.9)
                        continue
                    
                    break
                except:
                    font_size = int(font_size * 0.9)
            
            # Final render
            font = ImageFont.truetype(self.font_path, font_size)
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (self.size - text_width) // 2 - bbox[0]
            y = (self.size - text_height) // 2 - bbox[1]
            
            draw.text((x, y), char, font=font, fill=0)
            
            # Convert to boolean array (True = stroke, False = background)
            bitmap_array = np.array(img)
            bitmap_bool = (bitmap_array < 128)  # Boolean array, 1 byte per pixel
            
            return bitmap_bool
            
        except Exception as e:
            print(f"✗ Failed to render '{char}': {e}")
            return None
    
    def generate_all(self, save_checkpoint_every=1000):
        """Generate ALL Traditional Chinese characters and save as NPZ"""
        print("="*60)
        print("GENERATING ALL TRADITIONAL CHINESE CHARACTERS")
        print("="*60)
        print(f"Size: {self.size}x{self.size} pixels")
        print(f"Format: Boolean array (1 byte per pixel)")
        print(f"Total characters: {self.total_chars:,}")
        print(f"Output file: {self.output_file}")
        print(f"Estimated time: 2-5 minutes")
        print(f"Estimated file size: ~{self.total_chars * self.size * self.size / (1024*1024):.1f} MB")
        print("="*60)
        
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return None
        
        # Store bitmaps and characters
        bitmaps_list = []
        characters_list = []
        
        start_time = time.time()
        
        for i, code_point in enumerate(range(self.start_code, self.end_code + 1)):
            char = chr(code_point)
            
            # Progress indicator
            if i % 500 == 0:
                elapsed = time.time() - start_time
                progress = (i + 1) / self.total_chars * 100
                chars_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (self.total_chars - i - 1) / chars_per_sec if chars_per_sec > 0 else 0
                
                print(f"Progress: {progress:.1f}% ({i+1:,}/{self.total_chars:,}) | "
                      f"Speed: {chars_per_sec:.1f} chars/sec | "
                      f"ETA: {remaining/60:.1f} min")
            
            # Render character
            bitmap = self.render_character(char)
            if bitmap is not None:
                bitmaps_list.append(bitmap)
                characters_list.append(char)
            
            # Optional: Save checkpoint
            if save_checkpoint_every and (i + 1) % save_checkpoint_every == 0:
                print(f"  💾 Saving checkpoint at {i+1:,} characters...")
                # Convert to numpy arrays
                temp_bitmaps = np.array(bitmaps_list, dtype=bool)
                temp_chars = np.array(characters_list)
                np.savez_compressed(f'{self.output_file}.tmp', 
                                   bitmaps=temp_bitmaps, 
                                   chars=temp_chars)
        
        # Convert to numpy arrays
        print("\n💾 Converting to numpy arrays...")
        all_bitmaps = np.array(bitmaps_list, dtype=bool)
        all_chars = np.array(characters_list)
        
        # Save final NPZ file
        print(f"💾 Saving to {self.output_file}...")
        np.savez_compressed(self.output_file, 
                           bitmaps=all_bitmaps, 
                           chars=all_chars,
                           size=self.size)
        
        elapsed = time.time() - start_time
        file_size = os.path.getsize(self.output_file) / (1024 * 1024)
        
        print("\n" + "="*60)
        print("✓ COMPLETE!")
        print("="*60)
        print(f"✓ Characters generated: {len(all_chars):,}")
        print(f"✓ Bitmap shape: {all_bitmaps.shape}")
        print(f"✓ Memory usage (boolean array): {all_bitmaps.nbytes / (1024*1024):.1f} MB")
        print(f"✓ Compressed file size: {file_size:.1f} MB")
        print(f"✓ Time taken: {elapsed/60:.1f} minutes")
        print(f"✓ File saved: {self.output_file}")
        print("="*60)
        
        return all_bitmaps, all_chars
    
    def generate_test(self, count=10):
        """Generate first N characters for testing"""
        print(f"\n📝 TEST MODE: Generating first {count} characters")
        print("-"*60)
        
        bitmaps_list = []
        characters_list = []
        
        for i in range(count):
            code_point = self.start_code + i
            char = chr(code_point)
            print(f"  Rendering {i+1}/{count}: '{char}' (U+{code_point:04X})", end=' ')
            
            bitmap = self.render_character(char)
            if bitmap is not None:
                bitmaps_list.append(bitmap)
                characters_list.append(char)
                
                # Show utilization
                non_zero = np.sum(bitmap)
                utilization = (non_zero / (self.size * self.size)) * 100
                print(f"✓ {utilization:.1f}% filled")
            else:
                print("✗ Failed")
        
        # Save test file
        test_file = 'test_chinese_bitmaps.npz'
        test_bitmaps = np.array(bitmaps_list, dtype=bool)
        test_chars = np.array(characters_list)
        np.savez_compressed(test_file, bitmaps=test_bitmaps, chars=test_chars, size=self.size)
        
        print(f"\n✓ Test file saved: {test_file}")
        print(f"✓ Characters: {len(test_chars)}")
        print(f"✓ File size: {os.path.getsize(test_file) / 1024:.1f} KB")
        
        return test_bitmaps, test_chars


# ========== LOADER FUNCTION ==========

def load_chinese_bitmaps(filepath='chinese_bitmaps.npz'):
    """Load the NPZ file and return bitmap lookup function"""
    data = np.load(filepath, allow_pickle=True)
    bitmaps = data['bitmaps']  # Boolean array
    chars = data['chars']      # Character array
    size = data['size']        # Bitmap size
    
    print(f"✓ Loaded {len(chars)} characters")
    print(f"✓ Bitmap size: {size}x{size}")
    print(f"✓ Memory: {bitmaps.nbytes / (1024*1024):.1f} MB (uncompressed)")
    
    # Create a fast lookup dictionary
    char_to_index = {char: idx for idx, char in enumerate(chars)}
    
    def get_bitmap(char):
        """Get bitmap for a character"""
        if char in char_to_index:
            return bitmaps[char_to_index[char]]
        return None
    
    return get_bitmap, bitmaps, chars, char_to_index


# ========== QUICK ACCESS EXAMPLE ==========

def test_loading():
    """Example of how to load and use the bitmaps"""
    # Load the NPZ file
    get_bitmap, all_bitmaps, chars, char_to_index = load_chinese_bitmaps('chinese_bitmaps.npz')
    
    # Get specific character
    zhong = get_bitmap('中')
    if zhong is not None:
        print(f"\n'中' bitmap shape: {zhong.shape}")
        print(f"First 5 rows (first 10 columns):")
        for i in range(5):
            row = zhong[i][:10]
            row_str = ''.join(['█' if p else '·' for p in row])
            print(f"  {row_str}")
    
    # Access by index (faster)
    idx = char_to_index['國']
    guo = all_bitmaps[idx]
    print(f"\n'國' uses {np.sum(guo)} pixels")


# ========== MAIN ==========

if __name__ == "__main__":
    print("="*60)
    print("TRADITIONAL CHINESE BITMAP GENERATOR")
    print(f"Boolean NPZ Format - 1 byte per pixel")
    print("="*60)
    
    # Initialize generator
    generator = TraditionalChineseBitmapToNPZ(
        size=40,  # Change to 32 if you want 32x32
        output_file='chinese_bitmaps_40x40.npz'
    )
    
    # Test with first 10 characters
    print("\n🔧 Running test first...")
    test_bitmaps, test_chars = generator.generate_test(count=10)
    
    # Show sample bitmap
    if len(test_bitmaps) > 0:
        print(f"\n📊 Sample bitmap for '{test_chars[0]}':")
        bitmap = test_bitmaps[0]
        for i in range(min(40, len(bitmap))):
            row = bitmap[i][:40]
            row_str = ''.join(['█' if p else '·' for p in row])
            print(f"  {row_str}")
        
        print(f"\n💾 Memory: {test_bitmaps.nbytes / 1024:.1f} KB")
    
    # Generate ALL characters (uncomment when ready)
    print("\n" + "="*60)
    print("⚠️  To generate ALL 20,992 characters:")
    print("   Uncomment the line below and run again")
    print("="*60)
    
    # UNCOMMENT THIS LINE WHEN READY:
    all_bitmaps, all_chars = generator.generate_all()
    test_loading()