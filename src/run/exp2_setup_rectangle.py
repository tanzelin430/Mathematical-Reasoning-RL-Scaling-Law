
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Configuration
CONFIG = {
    # 'epochs': [1, 2, 3, 4, 5, 6, 7],
    'epochs': [1, 2, 5, 20, 25, 50, 100],
    'colors': {
        'background': 'white',
        # 'bar_edge': '#440154', # for inferno
        'bar_edge': '#08519C', # for YlGnBu_r
        'text': 'black',
        # 'cmap': 'inferno',  # Gradient colormap - purple to pink to yellow
        'cmap': 'YlGnBu_r',  # Gradient colormap - purple to pink to yellow
        'repeat_gradient': True,  # True: each segment has full gradient, False: continuous gradient
        'min_intensity': 0.1,  # Minimum intensity for the last epoch (0.0 to 1.0)
        'decay_rate': 0.65,    # Exponential decay rate (0.0-1.0, lower=faster decay)
        'gradient_direction': 'light_to_dark',  # 'light_to_dark' or 'dark_to_light'
        'decay_affects': 'dark'  # Which intensity to decay: 'light', 'dark', or 'both'
    },
    'dimensions': {
        'width': 0.88,
        'height': 0.05,
        'gap': 0.06,  # Reduced gap between rows # gap >= height
        'edge_width': 0.1
    }
}

def draw_simple_segmented_bar(ax, y, n_segments, config, epoch_index, total_epochs):
    """Draw a segmented bar with gradient and white separators"""
    width = config['dimensions']['width']
    height = config['dimensions']['height']
    seg_w = width / n_segments
    cmap = config['colors'].get('cmap', 'magma_r')
    repeat_gradient = config['colors'].get('repeat_gradient', True)
    
    # Calculate gradient intensity based on epoch position
    # First epoch (index 0) has full intensity (0 to 1)
    # Last epoch has reduced intensity
    max_intensity = 1.0
    min_intensity = config['colors'].get('min_intensity', 0.1)  # Configurable minimum intensity
    decay_power = config['colors'].get('decay_power', 2.0)  # Power for non-linear decay
    
    # Non-linear decrease in maximum intensity (faster decay)
    # Using exponential decay for more dramatic early reduction
    decay_rate = config['colors'].get('decay_rate', 0.7)  # How much to retain each step
    gradient_direction = config['colors'].get('gradient_direction', 'light_to_dark')
    decay_affects = config['colors'].get('decay_affects', 'both')
    
    if epoch_index == 0:
        # First epoch always full intensity for both sides
        light_intensity = max_intensity
        dark_intensity = 0
    else:
        # Calculate decay based on what should be affected
        decay_factor = decay_rate ** epoch_index
        
        if decay_affects == 'light':
            # Only light side gets lighter (affected by decay)
            light_intensity = max_intensity * decay_factor
            light_intensity = max(light_intensity, min_intensity)
            dark_intensity = 0  # Dark side stays at 0
        elif decay_affects == 'dark':
            # Only dark side gets lighter (affected by decay) 
            light_intensity = max_intensity  # Light side stays at max
            # Dark side: starts at 0, becomes lighter with decay (higher values = lighter)
            dark_intensity = (1 - decay_factor) * max_intensity  # Dark becomes lighter
            dark_intensity = min(dark_intensity, max_intensity - min_intensity)
        else:  # decay_affects == 'both'
            # Both sides affected equally
            light_intensity = max_intensity * decay_factor
            light_intensity = max(light_intensity, min_intensity)
            dark_intensity = 0
    
    # Draw each segment with gradient
    for i in range(n_segments):
        x0 = i * seg_w
        x1 = x0 + seg_w
        
        # Create gradient for this segment based on direction and decay settings
        W, H = 400, 60
        if repeat_gradient:
            # Each segment has gradient based on direction setting
            if gradient_direction == 'light_to_dark':
                # Left to right: light to dark
                start_intensity = light_intensity
                end_intensity = dark_intensity
            else:  # gradient_direction == 'dark_to_light'
                # Left to right: dark to light
                start_intensity = dark_intensity
                end_intensity = light_intensity
            
            grad = np.tile(np.linspace(start_intensity, end_intensity, W), (H, 1))
        else:
            # Continuous gradient across all segments with reduced intensity
            if gradient_direction == 'light_to_dark':
                start_val = ((n_segments - i) / n_segments) * light_intensity
                end_val = ((n_segments - i - 1) / n_segments) * light_intensity
            else:
                start_val = ((i + 1) / n_segments) * light_intensity  
                end_val = (i / n_segments) * light_intensity
            grad = np.tile(np.linspace(start_val, end_val, W), (H, 1))
        
        # Draw gradient with controlled intensity
        ax.imshow(grad, extent=[x0, x1, y - height/2, y + height/2],
                  origin='lower', cmap=cmap, aspect='auto', interpolation='bicubic',
                  vmin=0, vmax=1)  # Ensure consistent scaling
        
        # Add white separator between segments
        if i < n_segments - 1:  # Don't add separator after last segment
            sep_x = x1
            # Draw white line exactly at the segment boundary
            ax.plot([sep_x, sep_x], [y - height/2, y + height/2], 
                   color='white', linewidth=config['dimensions']['edge_width'])
    
    # Add outer border
    border_rect = Rectangle((0, y - height/2), width, height,
                           fill=False, 
                           edgecolor=config['colors']['bar_edge'],
                           linewidth=config['dimensions']['edge_width'])
    ax.add_patch(border_rect)

def make_figure(config=CONFIG):
    """Create the experimental setup figure with configurable parameters"""
    # Adjust figure size to reduce whitespace
    fig = plt.figure(figsize=(8, len(config['epochs']) * 0.8), dpi=160)
    ax = plt.gca()
    
    # Set background
    ax.set_facecolor(config['colors']['background'])
    fig.patch.set_facecolor(config['colors']['background'])
    
    # Calculate positions - start from top and work down
    total_height = len(config['epochs']) * config['dimensions']['gap']
    start_y = 1 - 0.1  # Small top margin
    
    for i, ep in enumerate(config['epochs']):
        y = start_y - i * config['dimensions']['gap']
        
        # Draw the segmented bar with progressive intensity
        draw_simple_segmented_bar(ax, y, ep, config, i, len(config['epochs']))
        
        # Add epoch label
        ax.text(config['dimensions']['width']/2, y, 
                # f"{ep} EPOCH" + ("S" if ep > 1 else ""), 
                f"τ = {ep}", 
                color=config['colors']['text'],
                fontsize=12, fontweight='bold', 
                ha='center', va='center')
    
    # Set limits with minimal margins
    ax.set_xlim(-0.02, config['dimensions']['width'] + 0.02)
    ax.set_ylim(start_y - len(config['epochs']) * config['dimensions']['gap'], start_y + 0.05)
    ax.axis('off')
    
    filepath = "outputs/exp2_tau_rectangles.pdf"
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', 
               facecolor=fig.get_facecolor(), dpi=300)
    print(f"Saved {filepath}")
    
    # plt.show()


# Additional color scheme examples
def make_purple_to_red_scheme():
    """Purple to red gradient scheme (similar to rainbow)"""
    purple_red_config = CONFIG.copy()
    purple_red_config['colors'] = CONFIG['colors'].copy()
    purple_red_config['colors']['cmap'] = 'plasma'  # Purple -> Pink -> Yellow
    purple_red_config['colors']['background'] = 'white'
    purple_red_config['colors']['bar_edge'] = '#8A2BE2'
    purple_red_config['colors']['text'] = '#4B0082'
    make_figure(purple_red_config)

def make_spectral_scheme():
    """Spectral rainbow-like scheme"""
    spectral_config = CONFIG.copy()
    spectral_config['colors'] = CONFIG['colors'].copy()
    spectral_config['colors']['cmap'] = 'Spectral_r'  # Red -> Orange -> Yellow -> Green -> Blue -> Purple
    spectral_config['colors']['background'] = 'white'
    spectral_config['colors']['bar_edge'] = '#8B0000'
    spectral_config['colors']['text'] = '#8B0000'
    make_figure(spectral_config)

def make_cool_warm_scheme():
    """Cool to warm scheme"""
    cool_warm_config = CONFIG.copy()
    cool_warm_config['colors'] = CONFIG['colors'].copy()
    cool_warm_config['colors']['cmap'] = 'coolwarm'  # Blue -> White -> Red
    cool_warm_config['colors']['background'] = 'white'
    cool_warm_config['colors']['bar_edge'] = '#B22222'
    cool_warm_config['colors']['text'] = '#000080'
    make_figure(cool_warm_config)

def make_inferno_scheme():
    """Inferno scheme - dark purple to bright yellow"""
    inferno_config = CONFIG.copy()
    inferno_config['colors'] = CONFIG['colors'].copy()
    inferno_config['colors']['cmap'] = 'inferno'  # Black -> Purple -> Red -> Orange -> Yellow
    inferno_config['colors']['background'] = 'white'
    inferno_config['colors']['bar_edge'] = '#800080'
    inferno_config['colors']['text'] = '#4B0082'
    make_figure(inferno_config)

def make_light_decay_demo():
    """Demo: Decay affects only light side (gets lighter over epochs)"""
    light_decay_config = CONFIG.copy()
    light_decay_config['colors'] = CONFIG['colors'].copy()
    light_decay_config['colors']['decay_affects'] = 'light'
    make_figure(light_decay_config)

def make_dark_decay_demo():
    """Demo: Decay affects only dark side (gets lighter over epochs)"""
    dark_decay_config = CONFIG.copy()
    dark_decay_config['colors'] = CONFIG['colors'].copy()
    dark_decay_config['colors']['decay_affects'] = 'dark'
    make_figure(dark_decay_config)

def make_both_decay_demo():
    """Demo: Decay affects both sides (current default)"""
    both_decay_config = CONFIG.copy()
    both_decay_config['colors'] = CONFIG['colors'].copy()
    both_decay_config['colors']['decay_affects'] = 'both'
    make_figure(both_decay_config)

def make_reversed_gradient_demo():
    """Demo: Reversed gradient direction (dark to light)"""
    reversed_config = CONFIG.copy()
    reversed_config['colors'] = CONFIG['colors'].copy()
    reversed_config['colors']['gradient_direction'] = 'dark_to_light'
    make_figure(reversed_config)

if __name__ == "__main__":
    make_figure()  # Use default configuration
    
    # Uncomment to try different decay modes:
    # make_light_decay_demo()      # Decay affects light side only
    # make_dark_decay_demo()       # Decay affects dark side only  
    # make_both_decay_demo()       # Decay affects both sides (default)
    # make_reversed_gradient_demo() # Reverse gradient direction (dark→light)
    
    # Uncomment to try different purple-to-red color schemes:
    # make_purple_to_red_scheme()  # Purple -> Pink -> Yellow (recommended!)
    # make_spectral_scheme()       # Full rainbow spectrum  
    # make_cool_warm_scheme()      # Blue -> White -> Red
    # make_inferno_scheme()        # Dark purple -> Red -> Yellow