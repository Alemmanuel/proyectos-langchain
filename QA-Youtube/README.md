# YouTube Q&A App with LangChain & Gemini

Welcome to the **Ultimate YouTube Q&A Machine**! This project lets you ask questions about any YouTube video and get instant, AI-powered answers. Powered by LangChain, Gemini, and Streamlit, it's like having a super-intelligent video assistant at your fingertips.

## ✨ Features
- 🎥 **Analyze any YouTube video:** Paste a link and get deep, AI-driven insights.
- 🤖 **Ask questions and get smart answers:** The AI understands the video and gives you context-aware responses.
- 🚀 **Fast, beautiful Streamlit interface:** Modern UI for a smooth experience.
- 🧠 **Embeddings and retrieval:** Deep video understanding for accurate answers.
- 🔒 **Secure API integration:** Your API keys are safe with dotenv.
- 🛠️ **Easy to extend:** Add new features, models, or UI tweaks with minimal effort.

## 📦 Installation & Usage
1. **Clone this repo:**
   ```bash
   git clone https://github.com/your-username/youtube-qa-langchain-gemini.git
   cd youtube-qa-langchain-gemini
   ```
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Add your Google API Key:**
   Create a `.env` file and add:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```
5. **Paste a YouTube link, process it, and start asking questions!**

## 🤩 Why This Project?
Because searching through videos is old school. This is the future—ask, and the AI answers. Try it and be amazed!

## 🧩 Project Structure
- `app.py` — The main Streamlit app
- `requirements.txt` — All dependencies
- `.env` — Your secret API keys (never share this!)
- `README.md` — You’re reading it!

## 💡 Customization Ideas
- Add support for playlists or channels
- Integrate with other video platforms
- Build a multi-user Q&A system
- Add video summarization or highlights

## 🌟 Credits & Inspiration
- Built with [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/), and [Google Gemini](https://ai.google.dev/)
- Inspired by the dream of making video knowledge instantly accessible