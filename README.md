# CarScan AI

An AI-powered vehicle damage detection and repair cost estimation platform with a React Native mobile app and FastAPI backend.

## Overview

CarScan AI automates the insurance adjustment workflow by combining:
- **Computer Vision** (YOLOv8) for real-time car damage detection from uploaded images
- **AI Multi-Agent Pipeline** (LangChain + LangGraph) for intelligent repair cost estimation with live web search
- **PDF Report Generation** for professional damage reports and cost estimates
- **Mobile-First Design** via Expo/React Native with tab navigation

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CarScan AI                       │
├──────────────────────┬──────────────────────────────┤
│     Backend          │      Mobile App              │
│    (FastAPI)         │   (Expo / React Native)      │
│                      │                              │
│  • YOLOv8 Detection  │  • Dashboard                 │
│  • LangChain Agents  │  • Camera/Image Upload       │
│  • PDF Generation    │  • Damage Detection w/ AI    │
│  • SSE Streaming     │  • Cost Estimation Pipeline  │
│                      │  • Settings & Configuration  │
└──────────┬───────────┴────────────┬─────────────────┘
           │                        │
           ▼                        ▼
   ┌──────────────┐         ┌──────────────┐
   │  Models      │         │  Components  │
   │  (best.pt)   │         │  • Buttons   │
   └──────────────┘         │  • Cards     │
                            │  • Badges    │
                            │  • Inputs    │
                            │  • Progress  │
                            │  • Toasts    │
                            └──────────────┘
```

## Features

### Damage Detection
- Capture or upload car photos via mobile camera or gallery
- YOLOv8 model detects 8 damage types: doors, windows, headlights, mirrors, dents, hoods, bumpers, windshields
- Color-coded bounding box annotations with confidence scores
- Severity classification: Minor, Moderate, Critical
- Full-screen image viewer with toggle between original and annotated views

### Cost Estimation
- Upload damage report PDFs for automated cost analysis
- **Search Agent**: Uses Tavily API to find real-time part prices
- **Reader Agent**: Scrapes automotive pricing sites for detailed data
- **Writer Agent**: Generates itemized estimates with OEM and aftermarket pricing
- **Critic Agent**: Validates estimates for realism and flags unrealistic pricing
- Live pipeline progress tracker with expandable agent log

### Mobile UI
- Bottom tab navigation: Dashboard, Scan, Estimates, Settings
- Professional design system with teal accent (#00D4AA) and deep blue primary (#1E3A5F)
- Component library: Buttons (4 variants), Cards, InputFields, Badges, ProgressBars, StatCards, Toasts
- Animated transitions with react-native-reanimated
- Persistent settings with AsyncStorage

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python, FastAPI, Uvicorn |
| **CV Model** | YOLOv8 (Ultralytics), OpenCV |
| **AI Agents** | LangChain, LangGraph, Groq (Llama 3.3) / Google Gemini |
| **Search** | Tavily API, BeautifulSoup |
| **PDF** | ReportLab, pdfplumber |
| **Mobile** | Expo, React Native, TypeScript |
| **Navigation** | Expo Router (Tabs) |
| **Animations** | react-native-reanimated |
| **Storage** | @react-native-async-storage |

## Project Structure

```
Proj-Main-Backup/
├── main.py                 # FastAPI backend
├── cost_agents.py          # LangChain agents (search, reader, writer, critic)
├── tools.py                # Tavily search and URL scraping tools
├── requirements.txt        # Python dependencies
├── .env                    # API keys
│
├── models/
│   └── best.pt             # YOLOv8 detection model
│
├── reports/                # Generated PDFs
│
├── static/                 # Legacy HTML (served by FastAPI fallback)
│   └── index.html
│
└── mobile/                 # React Native app
    ├── app/
    │   ├── _layout.tsx     # Tab navigation root
    │   ├── index.tsx       # Dashboard screen
    │   ├── detect.tsx      # Scan & detection screen
    │   ├── estimate.tsx    # Cost estimation screen
    │   └── settings.tsx    # Settings screen
    ├── components/
    │   ├── Button.tsx      # 4-variant button component
    │   ├── Card.tsx        # Card with optional header
    │   ├── InputField.tsx  # Labeled text input
    │   ├── Badge.tsx       # Color-coded status pills
    │   ├── ProgressBar.tsx # Progress indicator
    │   ├── StatCard.tsx    # Dashboard stat cards
    │   └── ToastNotifier.tsx
    ├── constants/
    │   ├── theme.ts        # Design tokens (colors, fonts, spacing)
    │   └── api.ts          # API endpoint configuration
    └── hooks/
        └── useToast.ts     # Toast notification system
```

## Setup & Installation

### Backend

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate    # Windows
   source venv/bin/activate   # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure `.env`:**
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Start server:**
   ```bash
   python main.py
   ```
   Access at `http://localhost:8000`

### Mobile App

1. **Navigate and install:**
   ```bash
   cd mobile
   npm install
   ```

2. **Configure API URL** in `mobile/constants/api.ts`:
   ```ts
   // Windows PC IP (run ipconfig)
   const BASE_URL = 'http://192.168.x.x:8000';
   // Android Emulator
   const BASE_URL = 'http://10.0.2.2:8000';
   ```

3. **Start Expo:**
   ```bash
   npm start
   ```
   Scan QR code with Expo Go or run:
   ```bash
   npm run android
   ```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/detect` | Upload image + vehicle info, return detections |
| `POST/GET` | `/generate-report` | Generate PDF damage report |
| `POST` | `/estimate-cost` | Upload PDFs to start cost pipeline |
| `GET` | `/estimate-stream/{id}` | SSE stream for pipeline progress |
| `POST/GET` | `/download-cost-report/{id}` | Download cost estimate PDF |

## Detection Classes

| Class | Color |
|-------|-------|
| Damaged Door | Red |
| Damaged Window | Blue |
| Damaged Headlight | Orange |
| Damaged Mirror | Cyan |
| Dent | Green |
| Damaged Hood | Purple |
| Damaged Bumper | Yellow |
| Damaged Wind Shield | Purple |

## Design System

### Colors
- **Primary**: Deep Blue `#1E3A5F` — trust, professionalism
- **Accent**: Electric Teal `#00D4AA` — AI/tech, CTAs
- **Success**: `#22C55E` | **Warning**: `#F59E0B` | **Error**: `#EF4444`
- **Background**: `#FAFBFC` with white cards

### Components
- **Button**: Primary (solid teal), Secondary (outlined), Ghost (text), Destructive (red)
- **Card**: White background, 8px radius, subtle shadow, optional header with icon
- **Badge**: Severity pills (green→yellow→orange→red), status indicators
- **InputField**: Labeled inputs with error states and focus rings
- **ProgressBar**: Configurable height and color
- **StatCard**: Dashboard metric cards with icon, value, label
- **Toast**: Slide-in notifications with auto-dismiss (4s)
