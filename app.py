import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import warnings
import requests
from io import BytesIO
warnings.filterwarnings('ignore')

# ============================================
# MODEL ARCHITECTURE (Same as your training)
# ============================================

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64, use_batchnorm=False)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.enc5 = EncoderBlock(512, 512)
        self.enc6 = EncoderBlock(512, 512)
        self.enc7 = EncoderBlock(512, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.dec1 = DecoderBlock(512, 512, use_dropout=True)
        self.dec2 = DecoderBlock(1024, 512, use_dropout=True)
        self.dec3 = DecoderBlock(1024, 512, use_dropout=True)
        self.dec4 = DecoderBlock(1024, 512)
        self.dec5 = DecoderBlock(1024, 256)
        self.dec6 = DecoderBlock(512, 128)
        self.dec7 = DecoderBlock(256, 64)
        
        # Final output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        
        b = self.bottleneck(e7)
        
        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        
        return self.final(torch.cat([d7, e1], dim=1))

# ============================================
# HELPER FUNCTIONS
# ============================================

@st.cache_resource
def load_model(weights_url, device):
    """Load the generator model with trained weights from URL"""
    model = Generator(in_channels=3, out_channels=3)
    
    try:
        # Download weights from GitHub
        with st.spinner("Downloading model weights..."):
            response = requests.get(weights_url, timeout=30)
            response.raise_for_status()
            
            # Save temporarily
            temp_path = "/tmp/best_generator.pth"
            with open(temp_path, "wb") as f:
                f.write(response.content)
            
            # Load with proper handling
            state_dict = torch.load(temp_path, map_location=device, weights_only=False)
            
            # Handle DataParallel weights
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=True)
            model.to(device)
            model.eval()
            
            st.success("✅ Model loaded successfully!")
            return model
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Trying fallback method...")
        
        # Fallback: Try loading with weights_only=False
        try:
            state_dict = torch.load(temp_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model
        except Exception as e2:
            st.error(f"Fallback also failed: {str(e2)}")
            return None

def preprocess_image(image, target_size=256):
    """Preprocess input image for the model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    img_array = np.array(image).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

def postprocess_image(tensor):
    """Convert model output tensor to PIL image"""
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_array)

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(
    page_title="Sketch to Anime - Conditional GAN",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Sketch to Anime Generator")
st.markdown("### Conditional GAN (Pix2Pix) based Anime Generator")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    
    device = "cpu"  # Force CPU for compatibility
    st.info(f"🖥️ Using device: **{device.upper()}**")
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Upload a sketch OR choose a sample
    2. Click 'Generate Anime'
    3. Download your result!
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Tips")
    st.markdown("""
    - Use clear sketches with defined edges
    - Best results with face/sketch drawings
    """)

# Load model (from GitHub release)
weights_url = "https://github.com/Mustehsan-Nisar-Rao/Conditional-GAN/releases/download/v1.0/best_generator.pth"

generator = load_model(weights_url, device)

if generator is None:
    st.stop()

# Sample images
sample_images = {
    "Sample 1": "https://raw.githubusercontent.com/Mustehsan-Nisar-Rao/Conditional-GAN/main/1014091%20(1).png",
    "Sample 2": "https://raw.githubusercontent.com/Mustehsan-Nisar-Rao/Conditional-GAN/main/1014091.png",
    "Sample 3": "https://raw.githubusercontent.com/Mustehsan-Nisar-Rao/Conditional-GAN/main/1014092.png",
    "Sample 4": "https://raw.githubusercontent.com/Mustehsan-Nisar-Rao/Conditional-GAN/main/1014093.png",
    "Sample 5": "https://raw.githubusercontent.com/Mustehsan-Nisar-Rao/Conditional-GAN/main/1014094.png"
}

# Main content - Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Input")
    
    # Tab for upload vs sample
    tab1, tab2 = st.tabs(["🖼️ Upload Sketch", "🎨 Try Samples"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a sketch/image...",
            type=['png', 'jpg', 'jpeg', 'webp']
        )
        input_image = None
        if uploaded_file:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Your Sketch", use_container_width=True)
    
    with tab2:
        selected_sample = st.selectbox("Select a sample sketch:", list(sample_images.keys()))
        if st.button("Load Sample", use_container_width=True):
            try:
                response = requests.get(sample_images[selected_sample])
                input_image = Image.open(BytesIO(response.content))
                st.image(input_image, caption=f"Sample: {selected_sample}", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading sample: {e}")
    
    generate_btn = st.button("✨ Generate Anime", type="primary", use_container_width=True)

with col2:
    st.subheader("✨ Output")
    output_placeholder = st.empty()
    
    if generate_btn and input_image:
        with st.spinner("Generating anime image... 🎨"):
            try:
                # Preprocess
                input_tensor = preprocess_image(input_image).to(device)
                
                # Generate
                with torch.no_grad():
                    output_tensor = generator(input_tensor)
                
                # Postprocess
                output_image = postprocess_image(output_tensor)
                
                # Display
                output_placeholder.image(output_image, caption="Generated Anime", use_container_width=True)
                
                # Download button
                buf = io.BytesIO()
                output_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 Download Result",
                    data=byte_im,
                    file_name="generated_anime.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                st.error("Please check if the model weights are compatible or try a different sketch.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with 🧡 using Conditional GAN (Pix2Pix) | Model trained on Anime Dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)
