"""
Generate high-quality ISL Gesture Reference Images v2
Creates clean, modern reference images for gesture words using Pillow
with gradient backgrounds, clean typography, and professional styling
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'static', 'gestures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIZE = 256

# Modern dark color scheme
BG_TOP = (20, 25, 40)
BG_BOTTOM = (35, 45, 65)
SKIN = (198, 156, 109)
SKIN_DARK = (165, 125, 85)
SKIN_LIGHT = (220, 185, 145)
OUTLINE = (130, 95, 65)
ACCENT = (80, 170, 255)
ACCENT2 = (100, 220, 180)
TEXT_WHITE = (245, 248, 255)
TEXT_DIM = (170, 180, 200)
CARD_BG = (30, 38, 55)
CARD_BORDER = (55, 65, 85)

GESTURES = {
    'ASSISTANCE': ('Flat hand lifts\nclosed fist upward', 'fist_lift'),
    'BAD':        ('Flat hand moves\ndown from chin', 'palm_down'),
    'BECOME':     ('Both palms rotate\nto swap positions', 'rotate'),
    'COLLEGE':    ('C-shape near temple\nmoves forward', 'c_hand'),
    'FROM':       ('Index pulls back\nfrom other index', 'two_point'),
    'PAIN':       ('Index fingers point\nat each other, twist', 'two_point'),
    'PRAY':       ('Both palms pressed\ntogether (Namaste)', 'namaste'),
    'SECONDARY':  ('Two fingers (V)\nmove to second spot', 'v_hand'),
    'SKIN':       ('Pinch and pull\nback of hand', 'pinch'),
    'SMALL':      ('Both flat hands\nclose together', 'hands_close'),
    'SPECIFIC':   ('F-shape moves\ndown with emphasis', 'f_hand'),
    'STAND':      ('V-fingers stand\non flat palm', 'v_on_palm'),
    'TODAY':      ('Both hands drop\ndown, palms up', 'hands_drop'),
    'WARN':       ('Palm taps back\nof other hand', 'tap'),
    'WHICH':      ('Both thumbs up\nalternate up/down', 'thumbs'),
    'YOU':        ('Index finger\npoints at person', 'point'),
    # New 20 gestures
    'MOTHER':     ('Thumb touches chin\nfingers spread open', 'palm_down'),
    'FATHER':     ('Thumb touches forehead\nfingers spread open', 'fist_lift'),
    'BROTHER':    ('Both fists bump\ntogether forward', 'hands_close'),
    'SISTER':     ('Index fingers\ninterlock together', 'two_point'),
    'FRIEND':     ('Hooked index fingers\nlink together', 'pinch'),
    'TEACHER':    ('Flat palms push\nforward from head', 'hands_drop'),
    'FOOD':       ('Bunched fingertips\ntouch mouth', 'pinch'),
    'WATER':      ('W-hand taps\nchin twice', 'v_hand'),
    'HOUSE':      ('Fingertips touch\nforming roof shape', 'namaste'),
    'FAMILY':     ('F-hands circle\ntogether outward', 'f_hand'),
    'SORRY':      ('Fist circles on\nchest (A-hand)', 'fist_lift'),
    'PLEASE':     ('Flat palm circles\non chest area', 'palm_down'),
    'MONEY':      ('Bunched fingers\ntap flat palm', 'tap'),
    'TIME':       ('Index finger taps\nwrist (watch area)', 'point'),
    'MORNING':    ('Arm rises upward\npalm faces up', 'fist_lift'),
    'NIGHT':      ('Curved hand arcs\ndown over fist', 'c_hand'),
    'HAPPY':      ('Flat palms brush\nupward on chest', 'hands_drop'),
    'SAD':        ('Open hands slide\ndown in front of face', 'palm_down'),
    'THANKYOU':   ('Flat hand from chin\nmoves forward', 'tap'),
    'GOODBYE':    ('Open palm waves\nside to side', 'rotate'),
}


def draw_gradient_bg(img):
    """Draw a smooth vertical gradient background"""
    draw = ImageDraw.Draw(img)
    for y in range(SIZE):
        ratio = y / SIZE
        r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * ratio)
        g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * ratio)
        b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * ratio)
        draw.line([(0, y), (SIZE, y)], fill=(r, g, b))


def draw_hand(draw, cx, cy, gesture_type, s=1.0):
    """Draw hand illustration based on gesture type"""
    
    if gesture_type == 'fist_lift':
        # Fist on top
        draw.rounded_rectangle([(cx-20*s, cy-30*s), (cx+20*s, cy-5*s)], radius=int(8*s), fill=SKIN, outline=OUTLINE, width=1)
        for i in range(-12, 16, 8):
            draw.line([(cx+i*s, cy-25*s), (cx+i*s, cy-15*s)], fill=SKIN_DARK, width=1)
        draw.ellipse([(cx-24*s, cy-18*s), (cx-16*s, cy-8*s)], fill=SKIN_LIGHT)
        # Flat hand below
        draw.rounded_rectangle([(cx-25*s, cy+5*s), (cx+25*s, cy+18*s)], radius=int(4*s), fill=SKIN, outline=OUTLINE, width=1)
        # Arrow
        draw.line([(cx+30*s, cy+5*s), (cx+30*s, cy-25*s)], fill=ACCENT, width=2)
        draw.polygon([(cx+26*s, cy-20*s), (cx+34*s, cy-20*s), (cx+30*s, cy-30*s)], fill=ACCENT)
        
    elif gesture_type == 'palm_down':
        draw.rounded_rectangle([(cx-25*s, cy-8*s), (cx+25*s, cy+12*s)], radius=int(5*s), fill=SKIN, outline=OUTLINE, width=1)
        for x in [-16, -6, 4, 14]:
            draw.ellipse([(cx+x*s, cy-13*s), (cx+(x+8)*s, cy-5*s)], fill=SKIN_LIGHT)
        draw.line([(cx, cy+18*s), (cx, cy+38*s)], fill=ACCENT, width=2)
        draw.polygon([(cx-5*s, cy+33*s), (cx+5*s, cy+33*s), (cx, cy+40*s)], fill=ACCENT)
        
    elif gesture_type == 'rotate':
        draw.rounded_rectangle([(cx-40*s, cy-10*s), (cx-10*s, cy+8*s)], radius=int(4*s), fill=SKIN, outline=OUTLINE, width=1)
        draw.rounded_rectangle([(cx+10*s, cy-10*s), (cx+40*s, cy+8*s)], radius=int(4*s), fill=SKIN_LIGHT, outline=OUTLINE, width=1)
        draw.arc([(cx-15*s, cy-25*s), (cx+15*s, cy+25*s)], 200, 340, fill=ACCENT, width=2)
        draw.arc([(cx-15*s, cy-25*s), (cx+15*s, cy+25*s)], 20, 160, fill=ACCENT2, width=2)
        
    elif gesture_type == 'c_hand':
        draw.arc([(cx-25*s, cy-28*s), (cx+25*s, cy+28*s)], 40, 320, fill=SKIN, width=int(10*s))
        draw.ellipse([(cx+18*s, cy+8*s), (cx+28*s, cy+18*s)], fill=SKIN_LIGHT)
        draw.ellipse([(cx+18*s, cy-18*s), (cx+28*s, cy-8*s)], fill=SKIN_LIGHT)
        
    elif gesture_type == 'two_point':
        # Two pointing hands
        for dx, flip in [(-22, 1), (22, -1)]:
            bx = cx + dx*s
            draw.ellipse([(bx-14*s, cy-2*s), (bx+14*s, cy+20*s)], fill=SKIN, outline=OUTLINE, width=1)
            draw.line([(bx, cy-2*s), (bx, cy-35*s)], fill=SKIN, width=max(1,int(6*s)))
            draw.ellipse([(bx-3*s, cy-38*s), (bx+3*s, cy-32*s)], fill=SKIN_LIGHT)
            
    elif gesture_type == 'namaste':
        # Prayer/Namaste hands
        for dx in [-12, 12]:
            bx = cx + dx*s
            draw.ellipse([(bx-14*s, cy+2*s), (bx+14*s, cy+25*s)], fill=SKIN, outline=OUTLINE, width=1)
            fingers = [(-8, -35), (-3, -42), (3, -42), (8, -35)]
            for fx, fy in fingers:
                draw.line([(bx, cy+2*s), (bx+fx*s, cy+fy*s)], fill=SKIN, width=max(1,int(5*s)))
                draw.ellipse([(bx+fx*s-2*s, cy+fy*s-2*s), (bx+fx*s+2*s, cy+fy*s+2*s)], fill=SKIN_LIGHT)
                
    elif gesture_type == 'v_hand':
        draw.ellipse([(cx-14*s, cy+5*s), (cx+14*s, cy+25*s)], fill=SKIN, outline=OUTLINE, width=1)
        draw.line([(cx-4*s, cy+5*s), (cx-14*s, cy-32*s)], fill=SKIN, width=max(1,int(6*s)))
        draw.line([(cx+4*s, cy+5*s), (cx+14*s, cy-32*s)], fill=SKIN, width=max(1,int(6*s)))
        draw.ellipse([(cx-16*s, cy-35*s), (cx-10*s, cy-29*s)], fill=SKIN_LIGHT)
        draw.ellipse([(cx+10*s, cy-35*s), (cx+16*s, cy-29*s)], fill=SKIN_LIGHT)
        
    elif gesture_type == 'pinch':
        draw.ellipse([(cx-15*s, cy+2*s), (cx+15*s, cy+22*s)], fill=SKIN, outline=OUTLINE, width=1)
        draw.line([(cx-6*s, cy+2*s), (cx-2*s, cy-28*s)], fill=SKIN, width=max(1,int(5*s)))
        draw.line([(cx+6*s, cy+2*s), (cx+2*s, cy-28*s)], fill=SKIN, width=max(1,int(5*s)))
        draw.ellipse([(cx-4*s, cy-32*s), (cx+4*s, cy-26*s)], fill=ACCENT2)
        
    elif gesture_type == 'hands_close':
        for dx in [-28, 28]:
            bx = cx + dx*s
            draw.rounded_rectangle([(bx-18*s, cy-8*s), (bx+18*s, cy+10*s)], radius=int(4*s), fill=SKIN, outline=OUTLINE, width=1)
        draw.line([(cx-8*s, cy+15*s), (cx-2*s, cy+15*s)], fill=ACCENT, width=2)
        draw.polygon([(cx-4*s, cy+12*s), (cx-4*s, cy+18*s), (cx+2*s, cy+15*s)], fill=ACCENT)
        draw.line([(cx+8*s, cy+15*s), (cx+2*s, cy+15*s)], fill=ACCENT, width=2)
        draw.polygon([(cx+4*s, cy+12*s), (cx+4*s, cy+18*s), (cx-2*s, cy+15*s)], fill=ACCENT)
        
    elif gesture_type == 'f_hand':
        draw.ellipse([(cx-15*s, cy+2*s), (cx+15*s, cy+22*s)], fill=SKIN, outline=OUTLINE, width=1)
        draw.line([(cx-8*s, cy+2*s), (cx-2*s, cy-20*s)], fill=SKIN, width=max(1,int(5*s)))
        draw.line([(cx+2*s, cy+2*s), (cx-2*s, cy-20*s)], fill=SKIN, width=max(1,int(5*s)))
        draw.ellipse([(cx-5*s, cy-23*s), (cx+1*s, cy-17*s)], fill=SKIN_LIGHT)
        draw.line([(cx+3*s, cy+2*s), (cx+8*s, cy-30*s)], fill=SKIN, width=max(1,int(5*s)))
        draw.line([(cx+10*s, cy+2*s), (cx+16*s, cy-28*s)], fill=SKIN, width=max(1,int(5*s)))
        draw.line([(cx+16*s, cy+2*s), (cx+22*s, cy-22*s)], fill=SKIN, width=max(1,int(4*s)))
        
    elif gesture_type == 'v_on_palm':
        draw.ellipse([(cx-10*s, cy-20*s), (cx+10*s, cy+2*s)], fill=SKIN, outline=OUTLINE, width=1)
        draw.line([(cx-4*s, cy-20*s), (cx-8*s, cy-42*s)], fill=SKIN, width=max(1,int(5*s)))
        draw.line([(cx+4*s, cy-20*s), (cx+8*s, cy-42*s)], fill=SKIN, width=max(1,int(5*s)))
        draw.ellipse([(cx-10*s, cy-45*s), (cx-6*s, cy-39*s)], fill=SKIN_LIGHT)
        draw.ellipse([(cx+6*s, cy-45*s), (cx+10*s, cy-39*s)], fill=SKIN_LIGHT)
        draw.rounded_rectangle([(cx-22*s, cy+5*s), (cx+22*s, cy+18*s)], radius=int(4*s), fill=SKIN_LIGHT, outline=OUTLINE, width=1)
        
    elif gesture_type == 'hands_drop':
        for dx in [-22, 22]:
            bx = cx + dx*s
            draw.rounded_rectangle([(bx-16*s, cy-6*s), (bx+16*s, cy+10*s)], radius=int(4*s), fill=SKIN, outline=OUTLINE, width=1)
        draw.line([(cx-22*s, cy+16*s), (cx-22*s, cy+32*s)], fill=ACCENT, width=2)
        draw.polygon([(cx-26*s, cy+28*s), (cx-18*s, cy+28*s), (cx-22*s, cy+35*s)], fill=ACCENT)
        draw.line([(cx+22*s, cy+16*s), (cx+22*s, cy+32*s)], fill=ACCENT, width=2)
        draw.polygon([(cx+18*s, cy+28*s), (cx+26*s, cy+28*s), (cx+22*s, cy+35*s)], fill=ACCENT)
        
    elif gesture_type == 'tap':
        draw.rounded_rectangle([(cx-20*s, cy-22*s), (cx+20*s, cy-2*s)], radius=int(6*s), fill=SKIN, outline=OUTLINE, width=1)
        draw.rounded_rectangle([(cx-22*s, cy+5*s), (cx+22*s, cy+22*s)], radius=int(4*s), fill=SKIN_LIGHT, outline=OUTLINE, width=1)
        for i in range(3):
            x = cx + 26*s + i*6*s
            draw.line([(x, cy-5*s), (x+4*s, cy-10*s)], fill=ACCENT, width=1)
            draw.line([(x, cy+3*s), (x+4*s, cy+8*s)], fill=ACCENT, width=1)
            
    elif gesture_type == 'thumbs':
        for dx, dy in [(-25, -5), (25, 5)]:
            bx, by = cx + dx*s, cy + dy*s
            draw.rounded_rectangle([(bx-12*s, by-3*s), (bx+12*s, by+14*s)], radius=int(5*s), fill=SKIN, outline=OUTLINE, width=1)
            draw.line([(bx-6*s, by-3*s), (bx-6*s, by-25*s)], fill=SKIN, width=max(1,int(6*s)))
            draw.ellipse([(bx-9*s, by-28*s), (bx-3*s, by-22*s)], fill=SKIN_LIGHT)
            
    elif gesture_type == 'point':
        draw.ellipse([(cx-15*s, cy-2*s), (cx+15*s, cy+22*s)], fill=SKIN, outline=OUTLINE, width=1)
        for x in [-8, 0, 8]:
            x0 = min(cx+x*s, cx+(x+8)*s)
            x1 = max(cx+x*s, cx+(x+8)*s)
            draw.arc([(x0, cy-2*s), (x1, cy+10*s)], 180, 0, fill=SKIN_DARK, width=max(1,int(2*s)))
        draw.line([(cx, cy-2*s), (cx, cy-42*s)], fill=SKIN, width=max(1,int(6*s)))
        draw.ellipse([(cx-3*s, cy-45*s), (cx+3*s, cy-39*s)], fill=SKIN_LIGHT)
        # Arrow emphasizing forward direction
        draw.line([(cx+5*s, cy-42*s), (cx+20*s, cy-42*s)], fill=ACCENT, width=2)
        draw.polygon([(cx+16*s, cy-46*s), (cx+16*s, cy-38*s), (cx+22*s, cy-42*s)], fill=ACCENT)


def create_image(word, desc, gesture_type):
    """Create a professional gesture reference image"""
    img = Image.new('RGB', (SIZE, SIZE), BG_TOP)
    draw_gradient_bg(img)
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        desc_font = ImageFont.truetype("arial.ttf", 13)
        label_font = ImageFont.truetype("arial.ttf", 10)
    except:
        title_font = ImageFont.load_default()
        desc_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Title bar
    draw.rectangle([(0, 0), (SIZE, 40)], fill=(15, 20, 35))
    # Accent line under title
    for x in range(SIZE):
        ratio = x / SIZE
        r = int(ACCENT[0] + (ACCENT2[0] - ACCENT[0]) * ratio)
        g = int(ACCENT[1] + (ACCENT2[1] - ACCENT[1]) * ratio)
        b = int(ACCENT[2] + (ACCENT2[2] - ACCENT[2]) * ratio)
        draw.point((x, 40), fill=(r, g, b))
        draw.point((x, 41), fill=(r, g, b))
    
    # Title text centered
    bbox = draw.textbbox((0, 0), word, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((SIZE - tw) // 2, 8), word, fill=TEXT_WHITE, font=title_font)
    
    # Hand illustration
    draw_hand(draw, SIZE // 2, 115, gesture_type, s=1.3)
    
    # Description card at bottom
    card_y = 178
    draw.rounded_rectangle([(10, card_y), (SIZE-10, SIZE-10)], radius=8, fill=CARD_BG, outline=CARD_BORDER, width=1)
    
    # "ISL Sign" label
    draw.text((18, card_y + 5), "ISL Sign:", fill=ACCENT, font=label_font)
    
    # Description
    lines = desc.split('\n')
    for i, line in enumerate(lines):
        draw.text((18, card_y + 20 + i*16), line.strip(), fill=TEXT_DIM, font=desc_font)
    
    return img


def main():
    print("🎨 ISL Gesture Reference Image Generator v2")
    print("=" * 45)
    
    generated = 0
    for word, (desc, gesture_type) in GESTURES.items():
        out_path = os.path.join(OUTPUT_DIR, f'{word}.png')
        
        # Check file size - skip large AI-generated images
        if os.path.exists(out_path):
            fsize = os.path.getsize(out_path)
            if fsize > 50000:  # AI-generated images are >50KB
                print(f"  ⏭️  {word}.png (AI image, {fsize} bytes) - keeping")
                continue
        
        print(f"  🎨 Generating {word}.png...")
        img = create_image(word, desc, gesture_type)
        img.save(out_path, 'PNG', optimize=True)
        generated += 1
    
    print(f"\n✅ Done! Regenerated: {generated} images")
    print(f"📁 Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
