# openThasr 🇹🇭🎙️
### OpenThai 2.0 Streaming Speech-to-Text (ASR) Engine & Native MCP Server

[![Model](https://img.shields.io/badge/Model-iapp%2FOpenThai2.0--ASR--Streaming-orange.svg)](https://huggingface.co/iapp/OpenThai2.0-ASR-Streaming-ONNX)
[![MCP](https://img.shields.io/badge/MCP-Standard_2.0-purple.svg)](https://modelcontextprotocol.io/)
[![Hardware](https://img.shields.io/badge/CPU_Only-Ready-success.svg)](https://openvino.ai/)
[![License](https://img.shields.io/badge/License-OpenMDW_1.1-blue.svg)](https://openmdw.ai/license/1-1/)

**openThasr** เป็นเครื่องมือถอดเสียงพูดภาษาไทยและภาษาอังกฤษเป็นข้อความ (Automatic Speech Recognition - ASR) ความเร็วสูงพิเศษ ทำงานบน CPU ได้อย่างมีประสิทธิภาพในระดับ Real-Time ขับเคลื่อนด้วยโมเดลโอเพนซอร์ส [iapp/OpenThai2.0-ASR-Streaming](https://huggingface.co/iapp/OpenThai2.0-ASR-Streaming-ONNX) จาก **iApp Technology** ร่วมกับ **AIEAT** และ **OpenThai**

โปรเจกต์นี้มาพร้อม **Native Model Context Protocol (MCP) Server** เพื่อให้ AI Agents (Google Antigravity, Claude Desktop, Cursor ฯลฯ) สามารถเรียกใช้เครื่องมือถอดเสียงภาษาไทยจากไฟล์หรือ Base64 ได้ทันที

---

## 🌟 จุดเด่น (Highlights)
- ⚡ **Real-Time Speech-to-Text บน CPU:** ประมวลผลได้รวดเร็วกว่าเสียงพูดจริง (RTF < 1.0) บน Intel Core i5 ทั่วไป โดยไม่ต้องใช้ GPU
- 🪶 **ONNX Quantized (INT8):** ใช้โมเดลขนาดกะทัดรัด (~0.9 GB) กิน RAM น้อย (~1.5–2 GB)
- 🇹🇭 **ภาษาที่รองรับ:** ภาษาไทย (`th-TH`) และภาษาไทยปนอังกฤษ (`th-mix`)
- 🤖 **Native MCP Server:** เชื่อมต่อกับ AI Agent ได้โดยตรงผ่าน Stdio Transport
- 🖱️ **Drag & Drop:** บน Windows สามารถลากไฟล์เสียงมาวางทับที่ `run.bat` เพื่อถอดเสียงได้ทันที

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)
```
openThasr/
├── models/
│   ├── encoder.int8.onnx           # Conformer Audio Encoder (INT8)
│   ├── encoder.int8.onnx.data      # Encoder Weights
│   ├── decoder_joint.onnx          # RNN-T Decoder & Joint Network
│   └── tokens.txt                  # Vocabulary Tokens (14,988 tokens)
├── asr_engine.py                   # Core ASR Engine & Feature Extractor
├── asr.py                          # Command-line Interface (CLI)
├── mcp_server.py                   # MCP Server สำหรับ AI Agents
├── run.bat                         # Windows Batch File (Drag & Drop)
├── requirements.txt                # Python Dependencies
└── README.md                       # คู่มือการใช้งาน
```

---

## 🛠️ การใช้งาน (Usage)

### 1. ลากวางไฟล์เสียง (Drag & Drop)
ลากไฟล์ `.wav`, `.mp3`, `.flac`, หรือ `.ogg` มาปล่อยทับที่ [run.bat](file:///d:/project/openThasr/run.bat)

### 2. ใช้งานผ่าน Command Line (CLI)
```powershell
# ถอดเสียงจากไฟล์
.\.venv\Scripts\python.exe asr.py "path\to\audio.wav"

# ถอดเสียงภาษาไทยปนอังกฤษ (th-mix) และบันทึกลงไฟล์
.\.venv\Scripts\python.exe asr.py "meeting.mp3" --lang th-mix -o "transcript.txt"
```

---

## 🤖 การตั้งค่า MCP Server สำหรับ Google Antigravity
เพิ่มใน `~/.gemini/config/mcp_config.json`:
```json
{
  "mcpServers": {
    "openthai-asr": {
      "command": "D:\\project\\openThasr\\.venv\\Scripts\\python.exe",
      "args": [
        "D:\\project\\openThasr\\mcp_server.py"
      ]
    }
  }
}
```

### รายการเครื่องมือ MCP (Available Tools)
| Tool | Arguments | Description |
| :--- | :--- | :--- |
| **`thai_asr_file`** | `audio_path` *(string)*,<br>`lang` *(string, default="th-TH")*,<br>`blank_bias` *(float, default=-1.0)* | ถอดเสียงภาษาไทย/อังกฤษจากไฟล์เสียง (`.wav`, `.mp3`, `.flac` ฯลฯ) |
| **`thai_asr_base64`** | `audio_base64` *(string)*,<br>`lang` *(string, default="th-TH")* | ถอดเสียงจากข้อมูลเสียงที่เข้ารหัส Base64 |
| **`thai_asr_info`** | ไม่มี | แสดงข้อมูลสถาปัตยกรรมและรายละเอียดของโมเดล OpenThai ASR |
