import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
from skimage import img_as_ubyte

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
def load_model(weights_path, device):
    """Load the generator model with trained weights"""
    model = Generator(in_channels=3, out_channels=3)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location=device)
    
    # Handle DataParallel weights if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image, target_size=256):
    """Preprocess input image for the model"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize to [-1, 1]
    img_array = np.array(image).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

def postprocess_image(tensor):
    """Convert model output tensor to PIL image"""
    # Denormalize from [-1, 1] to [0, 255]
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and then to PIL
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_array)

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(
    page_title="Sketch to Anime - Conditional GAN",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Sketch to Anime Generator")
st.markdown("### Conditional GAN (Pix2Pix) based Anime Generator")
st.markdown("Upload a sketch or drawing, and the AI will generate an anime-style image!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    
    # Device selection
    device_option = st.radio(
        "Compute Device",
        options=["Auto", "CPU", "CUDA"],
        help="Auto will use GPU if available"
    )
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Upload a sketch or drawing
    2. Wait for processing
    3. See the magic happen!
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Tips")
    st.markdown("""
    - Use clear sketches with defined edges
    - Image will be resized to 256x256
    - Best results with face/sketch drawings
    """)

# Determine device
if device_option == "Auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif device_option == "CUDA":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"

st.info(f"🖥️ Using device: **{device.upper()}**")

# Load model
@st.cache_resource
def get_model():
    # Download from GitHub release if not present locally
    weights_path = "best_generator.pth"
    
    # For Streamlit Cloud, we need to download from release
    import os
    if not os.path.exists(weights_path):
        with st.spinner("Downloading model weights from GitHub release..."):
            import requests
            url = "https://github.com/Mustehsan-Nisar-Rao/Conditional-GAN/releases/download/v1.0/best_generator.pth"
            response = requests.get(url)
            with open(weights_path, "wb") as f:
                f.write(response.content)
    
    return load_model(weights_path, device)

try:
    generator = get_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a sketch/image...",
    type=['png', 'jpg', 'jpeg', 'webp'],
    help="Upload your sketch or drawing"
)

# Example images
with st.expander("📷 Try with example images"):
    st.markdown("Coming soon! You can upload your own sketches for now.")

# Main processing
if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    original_image = Image.open(uploaded_file)
    col1.markdown("### 📤 Input Sketch")
    col1.image(original_image, use_container_width=True)
    
    # Generate button
    if st.button("✨ Generate Anime", type="primary", use_container_width=True):
        with st.spinner("Generating anime image... 🎨"):
            try:
                # Preprocess
                input_tensor = preprocess_image(original_image).to(device)
                
                # Generate
                with torch.no_grad():
                    output_tensor = generator(input_tensor)
                
                # Postprocess
                output_image = postprocess_image(output_tensor)
                
                # Display result
                col2.markdown("### ✨ Generated Anime")
                col2.image(output_image, use_container_width=True)
                
                # Download button
                buf = io.BytesIO()
                output_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 Download Generated Image",
                    data=byte_im,
                    file_name="generated_anime.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                st.success("✅ Generation complete!")
                
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
                st.error("Please make sure your sketch is clear and try again.")

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
