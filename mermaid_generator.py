import mermaid
import io
from PIL import Image

def generate_mermaid_image(mermaid_code):
    try:
        chart = mermaid.Mermaid(code=mermaid_code)
        svg_data = chart.svg()
        
        # Convert SVG to PNG using CairoSVG (if available) or another method
        try:
            import cairosvg
            png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
            img = Image.open(io.BytesIO(png_data))
            return img
        except ImportError:
            print("CairoSVG is not installed. Using a basic text representation.")
            return None #Or return a text based representation if needed.

    except Exception as e:
        print(f"Error generating Mermaid image: {e}")
        return None

mermaid_code = """
graph TD
    subgraph "Master Process (Rank 0)"
        A[chess_gui.py] --> B[GUI Thread]
        A --> C[AI Move Thread]
        C --> D[master.py]
        D --> E[Task Distribution]
        D --> F[Result Collection]
    end
    
    subgraph "Worker Processes (Rank > 0)"
        G[worker.py] --> H[Task Reception]
        H --> I[chess_engine.py]
        I --> J[Minimax Search]
        J --> K[Board Evaluation]
        I --> L[Result Transmission]
    end
    
    subgraph "Shared Components"
        M[chess_engine.py] --> N[Minimax Algorithm]
        M --> O[Evaluation Function]
        P[utils.py] --> Q[Serialization]
        P --> R[Logging]
    end
    
    E -- "MPI Send (FEN, Depth, Time)" --> H
    L -- "MPI Receive (Score, Move)" --> F
    Q -- "Used by" --> E
    Q -- "Used by" --> L
    R -- "Used by" --> A
    R -- "Used by" --> D
    R -- "Used by" --> G
    R -- "Used by" --> I
    B --> O
    C --> N
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#bfb,stroke:#333,stroke-width:2px
    style M fill:#ffb,stroke:#333,stroke-width:2px
    style P fill:#fdd,stroke:#333,stroke-width:2px
"""

image = generate_mermaid_image(mermaid_code)

if image:
    image.save("mermaid_diagram.png")
    print("Mermaid diagram saved as mermaid_diagram.png")
else:
    print("Image generation failed.")