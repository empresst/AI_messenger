import os
import json
import re
import hashlib
import secrets
import base64
import uuid
import threading
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
from contextlib import asynccontextmanager

import pytz
import numpy as np
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# LangChain / FAISS / OpenAI
from cachetools import TTLCache
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from openai import AsyncOpenAI

# ---------------------
# Setup logging
# ---------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-twin-app")

# ---------------------
# NLTK & spaCy (hardened for Render)
# ---------------------
NLTK_DATA_DIR = os.getenv("NLTK_DATA", "/tmp/nltk_data")
os.environ["NLTK_DATA"] = NLTK_DATA_DIR
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

import nltk
nltk.download('wordnet', quiet=True, download_dir=NLTK_DATA_DIR)
nltk.download('punkt', quiet=True, download_dir=NLTK_DATA_DIR)
from nltk.corpus import wordnet

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    logger.warning("spaCy model 'en_core_web_sm' not available; using spacy.blank('en') fallback.")
    nlp = spacy.blank("en")

# ---------------------
# Env & constants
# ---------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
PUBLIC_UI_API_KEY = os.getenv("PUBLIC_UI_API_KEY", "your-secure-api-key")
PORT = int(os.getenv("PORT", "8000"))
SEED_DEMO = os.getenv("SEED_DEMO", "false").lower() == "true"
SESSION_TTL_MIN = int(os.getenv("SESSION_TTL_MIN", "4320"))  # 3 days

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI missing")

# ---------------------
# Globals (thread-safe)
# ---------------------
client: Optional[AsyncIOMotorClient] = None
openai_client: Optional[AsyncOpenAI] = None
faiss_store: Optional[FAISS] = None

db = None
users_col = None
conversations_col = None
journals_col = None
embeddings_col = None
personalities_col = None
errors_col = None
saved_greetings_col = None
greetings_cache_col = None
relationships_col = None
sessions_col = None

mongo_lock = threading.Lock()
openai_lock = threading.Lock()
faiss_lock = threading.Lock()

embedding_cache = TTLCache(maxsize=1000, ttl=3600)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
FAISS_DIR = "faiss_store_v1"
watcher_task: Optional[asyncio.Task] = None

def as_utc_aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=pytz.UTC)

# ---------------------
# Lazy init functions
# ---------------------
async def get_mongo_client() -> AsyncIOMotorClient:
    global client, db, users_col, conversations_col, journals_col, embeddings_col, personalities_col
    global errors_col, saved_greetings_col, greetings_cache_col, relationships_col, sessions_col
    with mongo_lock:
        if client is None:
            client = AsyncIOMotorClient(
                MONGODB_URI,
                tls=True,
                tlsAllowInvalidCertificates=True,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                tz_aware=True
            )
            db = client["LF"]
            users_col = db["users"]
            conversations_col = db["conversations"]
            journals_col = db["journal_entries"]
            embeddings_col = db["embeddings"]
            personalities_col = db["personalities"]
            errors_col = db["errors"]
            saved_greetings_col = db["saved_greetings"]
            greetings_cache_col = db["greetings"]
            relationships_col = db["relationships"]
            sessions_col = db["sessions"]

    # indexes
    await conversations_col.create_index([("user_id", 1), ("timestamp", -1)])
    await conversations_col.create_index([("speaker_id", 1), ("target_id", 1), ("timestamp", -1)])
    await conversations_col.create_index([("content", "text")])
    await journals_col.create_index([("user_id", 1), ("timestamp", -1)])
    await journals_col.create_index([("content", "text")])
    await embeddings_col.create_index([("item_id", 1), ("item_type", 1)])
    await personalities_col.create_index([("user_id", 1)])
    await errors_col.create_index([("timestamp", -1)])
    await saved_greetings_col.create_index([("target_id", 1), ("bot_role", 1), ("timestamp", -1)])
    await greetings_cache_col.create_index([("key", 1), ("timestamp", -1)])
    await relationships_col.create_index([("user_id", 1), ("other_user_id", 1)], unique=True)
    await sessions_col.create_index("expires_at", expireAfterSeconds=0)

    return client

async def get_openai_client() -> AsyncOpenAI:
    global openai_client
    with openai_lock:
        if openai_client is None:
            openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return openai_client

async def ensure_faiss_store():
    global faiss_store
    with faiss_lock:
        if faiss_store is None:
            if os.path.isdir(FAISS_DIR):
                try:
                    faiss_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                    return
                except Exception as e:
                    logger.warning(f"FAISS load failed: {e}. Rebuilding...")
    await initialize_faiss_store()

async def initialize_faiss_store():
    global faiss_store
    await get_mongo_client()
    with faiss_lock:
        if os.path.isdir(FAISS_DIR):
            try:
                faiss_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                return
            except Exception:
                pass

    emb_data = await embeddings_col.find().to_list(length=None)
    docs: List[Document] = []
    for emb in emb_data:
        try:
            item_id = emb.get("item_id")
            item_type = emb.get("item_type")
            if not item_id or not item_type:
                await embeddings_col.delete_one({"_id": emb["_id"]})
                continue
            col = conversations_col if item_type == "conversation" else journals_col
            id_field = "conversation_id" if item_type == "conversation" else "entry_id"
            base = await col.find_one({id_field: item_id})
            if not base:
                continue

            content = emb.get("content", base.get("content", ""))
            if not content:
                await embeddings_col.delete_one({"item_id": item_id, "item_type": item_type})
                continue
            owner_ids = emb.get("user_id", [])
            
            if item_type == "journal":
                # Ensure the saved journal actually belongs to the same owner list
                base_uids = base.get("user_id", [])
                if isinstance(base_uids, list) and not any(u in base_uids for u in owner_ids):
                    continue
                    
            metadata = {
                "item_id": item_id,
                "item_type": item_type,
                "user_id": emb.get("user_id", []),
                "speaker_id": emb.get("speaker_id"),
                "target_id": emb.get("target_id"),
                "speaker_name": emb.get("speaker_name"),
                "target_name": emb.get("target_name"),
                "timestamp": as_utc_aware(emb.get("timestamp"))
            }
            docs.append(Document(page_content=content, metadata=metadata))
        except Exception:
            await embeddings_col.delete_one({"_id": emb["_id"]})

    with faiss_lock:
        if docs:
            faiss_store = FAISS.from_documents(docs, embeddings)
        else:
            faiss_store = FAISS.from_texts(["empty"], embeddings)
        faiss_store.save_local(FAISS_DIR)

# ---------------------
# Security: session tokens
# ---------------------
async def create_session(user_id: str) -> str:
    await get_mongo_client()
    token = str(uuid.uuid4())
    now = datetime.now(pytz.UTC)
    await sessions_col.insert_one({
        "token": token,
        "user_id": user_id,
        "created_at": now,
        "expires_at": now + timedelta(minutes=SESSION_TTL_MIN)
    })
    return token

async def require_session(x_session_token: str = Header(...)) -> Dict[str, Any]:
    await get_mongo_client()
    sess = await sessions_col.find_one({"token": x_session_token})
    if not sess:
        raise HTTPException(status_code=401, detail="Invalid session")
    if as_utc_aware(sess["expires_at"]) < datetime.now(pytz.UTC):
        raise HTTPException(status_code=401, detail="Session expired")
    user = await users_col.find_one({"user_id": sess["user_id"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"token": x_session_token, "user": user}

# ---------------------
# Password hashing
# ---------------------
def hash_password(password: str) -> Dict[str, str]:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return {"salt": base64.b64encode(salt).decode(), "hash": base64.b64encode(dk).decode()}

def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    salt = base64.b64decode(salt_b64.encode())
    expected = base64.b64decode(hash_b64.encode())
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return secrets.compare_digest(dk, expected)

# ---------------------
# Connection manager (WebSockets)
# ---------------------
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}
        self.lock = asyncio.Lock()

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active[user_id] = websocket

    async def disconnect(self, user_id: str):
        async with self.lock:
            self.active.pop(user_id, None)

    async def send_to(self, user_id: str, data: dict):
        async with self.lock:
            ws = self.active.get(user_id)
        if ws:
            await ws.send_json(data)

    async def broadcast_presence(self):
        async with self.lock:
            online = list(self.active.keys())
            sockets = list(self.active.values())
        payload = {"type": "presence", "online": online}
        for ws in sockets:
            try:
                await ws.send_json(payload)
            except Exception:
                pass

manager = ConnectionManager()

# ---------------------
# FastAPI app (+ healthcheck)
# ---------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global watcher_task
    await initialize_db()
    watcher_task = asyncio.create_task(watch_collections())
    yield
    if watcher_task:
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
    if client:
        client.close()

app = FastAPI(title="Chatbot AI Twin API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def healthz():
    # Keep this independent of DB/OpenAI so Render health checks succeed
    return {"ok": True}

# ---------------------
# Pydantic models
# ---------------------
class MessageRequest(BaseModel):
    speaker_id: str
    target_id: str
    bot_role: Optional[str] = None
    user_input: str

class MessageResponse(BaseModel):
    response: str
    error: Optional[str] = None

class SignupRequest(BaseModel):
    username: str
    display_name: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class RelationshipSetRequest(BaseModel):
    other_user_id: str
    relation: str

class JournalAddRequest(BaseModel):
    content: str
    consent: bool

# ---------------------
# HTML UI (same as your enhanced version, with tiny fix: do not force bot_role)
# ---------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>AI Twin Chat</title>
<style>
* { box-sizing: border-box }
:root { --bg:#0f172a; --fg:#fff; --muted:#64748b; --border:#e2e8f0; --card:#f8fafc; }
body { margin:0; font-family: Inter, system-ui, Arial }
header { padding: 10px 16px; background:var(--bg); color:var(--fg); display:flex; justify-content:space-between; align-items:center }
main { display:flex; height: calc(100vh - 56px) }
#sidebar { width:360px; border-right:1px solid var(--border); padding:12px; overflow:auto }
#content { flex:1; display:flex; flex-direction:column }
section { margin-bottom:16px }
h3 { margin:8px 0 }
input, select, button, textarea { padding:10px; margin:6px 0; width:100%; font-size:14px }
textarea { resize:none; min-height:48px; max-height:160px; line-height:20px; }
button { cursor:pointer }
.user-item { padding:8px; border:1px solid var(--border); border-radius:8px; margin:6px 0; display:flex; gap:8px; align-items:center; justify-content:space-between; background:#fff }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#e2e8f0 }
.badge.online { background:#bbf7d0 }
.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#fde68a; color:#92400e; margin-left:6px }
.chatgrid { flex:1; display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:12px; padding:12px; overflow:auto }
.chatbox { border:1px solid var(--border); border-radius:10px; display:flex; flex-direction:column; min-height:360px; background:#fff }
.chatbox header { background:var(--card); color:#0f172a; font-weight:600; padding:8px 10px; display:flex; align-items:center; justify-content:space-between }
.header-right { display:flex; align-items:center; gap:6px }
.messages { flex:1; padding:10px; overflow:auto; font-size:14px; background:#fafafa }
.msg { margin:8px 0; padding:8px 10px; border-radius:8px; max-width:92%; box-shadow:0 1px 0 rgba(0,0,0,.04) }
.msg.you { background:#dbeafe; align-self:flex-end }
.msg.them { background:#dcfce7 }
.msg.ai { background:#fef3c7 }
.meta { font-size:11px; color:var(--muted); margin-top:4px }
.typing { font-size:12px; color:var(--muted); margin:6px 0 0 2px }
.actions { padding:8px; display:flex; gap:8px; border-top:1px solid var(--border); background:#fff }
.actions textarea { flex:1; border:1px solid var(--border); border-radius:8px; padding:10px 12px; }
.actions button { width:auto; padding:10px 14px; border-radius:8px; background:#0ea5e9; color:#fff; border:none }
.actions button:disabled { opacity:.6; cursor:not-allowed }
.small { font-size:12px; color:var(--muted) }
.form-card { border:1px solid var(--border); border-radius:10px; padding:12px; background:#fff }
.row { display:flex; gap:8px }
.row > * { flex:1 }
hr { border:0; border-top:1px solid var(--border); margin:12px 0 }
.list { border:1px solid var(--border); border-radius:10px; padding:8px; background:#fff; max-height:220px; overflow:auto }
.item { padding:6px 6px; border-bottom:1px dashed #e5e7eb }
.item:last-child { border-bottom:none }
.warn { color:#92400e; background:#fef3c7; padding:6px 8px; border-radius:8px; font-size:12px; }
</style>
</head>
<body>
<header>
  <div>AI Twin Chat</div>
  <div id="whoami" class="small"></div>
</header>
<main>
  <div id="sidebar">
    <section id="auth">
      <div class="form-card">
        <h3>Login</h3>
        <input id="loginUsername" placeholder="username"/>
        <input id="loginPassword" placeholder="password" type="password"/>
        <button id="loginBtn">Login</button>
      </div>
      <hr/>
      <div class="form-card">
        <h3>Sign up</h3>
        <input id="signupUsername" placeholder="username"/>
        <input id="signupDisplayName" placeholder="display name"/>
        <input id="signupPassword" placeholder="password" type="password"/>
        <button id="signupBtn">Create account</button>
      </div>
      <div style="margin-top:10px">
        <div class="small">Server key for UI (x-api-key)</div>
        <input id="gatewayKey" placeholder="x-api-key (server key)" />
      </div>
    </section>

    <section id="me" style="display:none">
      <div class="form-card">
        <h3>Me</h3>
        <div id="meInfo"></div>
        <div class="row">
          <label class="small" style="display:flex; align-items:center; gap:8px">
            <input type="checkbox" id="aiToggle"/>
            AI respond for me
          </label>
          <button id="logoutBtn" style="background:#ef4444">Logout</button>
        </div>
      </div>
    </section>

    <section id="users" style="display:none">
      <h3>Users</h3>
      <div id="usersList"></div>
    </section>

    <section id="rel" style="display:none">
      <h3>Set Relationship</h3>
      <div class="row">
        <select id="relOther"></select>
        <select id="relKind">
          <option>daughter</option><option>son</option><option>mother</option><option>father</option>
          <option>sister</option><option>brother</option><option>wife</option><option>husband</option>
          <option>friend</option>
        </select>
      </div>
      <button id="relSave">Save</button>
      <div id="relStatus" class="small"></div>
    </section>

    <!-- JOURNAL SECTION -->
    <section id="journal" style="display:none">
      <h3>Journal</h3>
      <div class="warn">Notes here become your private memory and may be used to personalize replies.</div>
      <textarea id="journalText" placeholder="Write a private note you'll want your AI Twin to remember…"></textarea>
      <label class="small" style="display:flex; gap:8px; align-items:center; margin-top:6px">
        <input type="checkbox" id="journalConsent"/> I understand and consent to this note being used to train my AI Twin.
      </label>
      <div class="row">
        <button id="saveJournalBtn">Save Journal</button>
        <button id="refreshJournalBtn" style="background:#64748b">Refresh</button>
      </div>
      <div id="journalStatus" class="small"></div>
      <div style="margin-top:8px" class="small">Recent entries</div>
      <div id="journalList" class="list"></div>
    </section>
  </div>

  <div id="content">
    <div class="chatgrid" id="chatGrid"></div>
  </div>
</main>

<script>
let API = location.origin;
let API_KEY = "";
let SESSION = "";
let ME = null;
let WS = null;
const chatBoxes = new Map(); // other_user_id -> {box, area, btn, typingEl, aiExpected}

function el(id){ return document.getElementById(id) }

function setAuthVisible(loggedIn){
  el('auth').style.display = loggedIn ? 'none' : 'block';
  el('me').style.display = loggedIn ? 'block' : 'none';
  el('users').style.display = loggedIn ? 'block' : 'none';
  el('rel').style.display = loggedIn ? 'block' : 'none';
  el('journal').style.display = loggedIn ? 'block' : 'none';
}

function autoresizeTA(ta){
  ta.style.height = 'auto';
  ta.style.height = Math.min(160, Math.max(48, ta.scrollHeight)) + 'px';
}

async function req(path, method='GET', body=null){
  const headers = {'Content-Type':'application/json'};
  if(API_KEY && API_KEY.toLowerCase()!=='disabled') headers['x-api-key']=API_KEY;
  if(SESSION) headers['x-session-token']=SESSION;
  const res = await fetch(API+path, {method, headers, body: body?JSON.stringify(body):undefined});
  if(!res.ok){ throw new Error(await res.text()) }
  return res.json();
}

function renderMe(){
  el('meInfo').innerHTML = `
    <div><b>${ME.display_name}</b> <span class="small">(@${ME.username})</span></div>
    <div class="small">user_id: ${ME.user_id}</div>
    <div class="small">AI: ${ME.ai_enabled ? 'ON' : 'OFF'}</div>
  `;
  el('whoami').innerText = `${ME.display_name} (@${ME.username})`;
  el('aiToggle').checked = !!ME.ai_enabled;
}

async function refreshUsers(){
  const data = await req('/users/list');
  const container = el('usersList');
  const relSel = el('relOther');
  container.innerHTML = '';
  relSel.innerHTML = '';
  (data.users || []).filter(u=>u.user_id!==ME.user_id).forEach(u=>{
     const div = document.createElement('div');
     div.className='user-item';
     const badge = `<span class="badge ${u.online?'online':''}">${u.online?'online':'offline'}</span>`;
     const ai = u.ai_enabled ? `<span class="pill">AI replies</span>` : '';
     div.innerHTML = `
       <div>
         <div><b>${u.display_name}</b> <span class="small">(@${u.username})</span> ${ai}</div>
         <div class="small">rel: ${u.relation || '-'}</div>
       </div>
       <div>
         ${badge}
         <button data-id="${u.user_id}">Chat</button>
       </div>`;
     div.querySelector('button').onclick=()=>openChat(u);
     container.appendChild(div);

     const opt = document.createElement('option');
     opt.value = u.user_id; opt.textContent = `${u.display_name} (@${u.username})`;
     relSel.appendChild(opt);
  });
}

function ensureBox(u){
  if(chatBoxes.has(u.user_id)) return chatBoxes.get(u.user_id);
  const div = document.createElement('div');
  div.className='chatbox';
  div.innerHTML = `
    <header>
      <div>${u.display_name} <span class="small">(@${u.username})</span></div>
      <div class="header-right">
        <span class="badge ${u.online?'online':''}" id="on_${u.user_id}">${u.online?'online':'offline'}</span>
        ${u.ai_enabled?'<span class="pill">AI replies</span>':''}
      </div>
    </header>
    <div class="messages" id="msg_${u.user_id}"></div>
    <div class="typing" id="typing_${u.user_id}" style="display:none">AI is typing…</div>
    <div class="actions">
      <textarea placeholder="Write a message…" id="inp_${u.user_id}"></textarea>
      <button id="send_${u.user_id}">Send</button>
    </div>
  `;
  el('chatGrid').appendChild(div);
  const area = el('inp_'+u.user_id);
  const btn = el('send_'+u.user_id);
  const typingEl = el('typing_'+u.user_id);
  area.addEventListener('input', ()=>autoresizeTA(area));
  area.addEventListener('keypress', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendTo(u.user_id); }});
  btn.onclick=()=>sendTo(u.user_id);
  chatBoxes.set(u.user_id, {box:div, area, btn, typingEl, aiExpected: !!u.ai_enabled});
  return chatBoxes.get(u.user_id);
}

async function openChat(u){
  const {area} = ensureBox(u);
  // load last 30 messages
  const res = await req(`/conversations/with/${u.user_id}?limit=30`);
  const msgs = res.messages || [];
  const pane = el('msg_'+u.user_id);
  pane.innerHTML='';
  msgs.forEach(m=>appendMsg(u.user_id, m));
  area.focus();
}

function appendMsg(other_id, m, localEcho=false){
  const pane = el('msg_'+other_id);
  if(!pane) return;
  const wrapper = document.createElement('div');
  const who = localEcho ? 'you' : (m.speaker_id===ME.user_id ? 'you' : (m.source==='ai_twin' ? 'ai' : 'them'));
  wrapper.className = `msg ${who}`;
  const when = new Date(m.timestamp).toLocaleString();
  wrapper.innerHTML = `${m.content}<div class="meta">${when}${localEcho?' • ✓ Sent':''}</div>`;
  pane.appendChild(wrapper);
  pane.scrollTop = pane.scrollHeight;
}

function showTyping(other_id, on){
  const elT = el('typing_'+other_id);
  if(!elT) return;
  elT.style.display = on ? 'block' : 'none';
}

async function sendTo(other_id){
  const ref = chatBoxes.get(other_id);
  if(!ref) return;
  const {area, btn, aiExpected} = ref;
  const text = (area.value || '').trim();
  if(!text) return;
  area.value=''; autoresizeTA(area);
  btn.disabled = true;

  // Local echo immediately
  appendMsg(other_id, {content:text, timestamp: new Date().toISOString(), speaker_id: ME.user_id, source:'human'}, true);

  // If we expect AI, show typing until reply arrives
  if(aiExpected) showTyping(other_id, true);

  // Prefer WS if connected, else HTTP
  if(WS && WS.readyState===1){
    WS.send(JSON.stringify({type:'chat', to: other_id, text}));
    // AI reply will arrive as WS "ai" event if enabled
    btn.disabled = false;
  }else{
    try{
      const res = await req('/send_message','POST', {
        speaker_id: ME.user_id,
        target_id: other_id,
        user_input: text
      });
      // If server returned an AI reply (HTTP path), append it
      if(res && res.response && res.response !== 'Sent.'){
        appendMsg(other_id, {content: res.response, timestamp: new Date().toISOString(), speaker_id: other_id, source:'ai_twin'});
      }
    }catch(e){
      appendMsg(other_id, {content: '⚠️ Failed to send: '+(e.message||e), timestamp: new Date().toISOString(), speaker_id: other_id, source:'system'});
    }finally{
      showTyping(other_id, false);
      btn.disabled = false;
    }
  }
}

function connectWS(){
  if(WS) try{ WS.close() }catch(e){}
  const qp = new URLSearchParams({ token: SESSION, user_id: ME.user_id });
  WS = new WebSocket(API.replace(/^http/,'ws')+'/ws?'+qp.toString());
  WS.onmessage = (ev)=>{
    try{
      const msg = JSON.parse(ev.data);
      if(msg.type==='presence'){
        // Update online badges quickly
        (msg.online||[]).forEach(uid=>{
          const b = document.getElementById('on_'+uid);
          if(b){ b.classList.add('online'); b.textContent='online'; }
        });
      }else if(msg.type==='chat'){
        appendMsg(msg.from, msg.payload);
      }else if(msg.type==='ai'){
        appendMsg(msg.from, msg.payload);
        showTyping(msg.from, false);
      }
    }catch(e){}
  };
}

/* ---- JOURNAL UI wiring ---- */
async function refreshJournal(){
  try{
    const res = await req('/journals/list','GET');
    const host = el('journalList');
    host.innerHTML = (res.entries||[]).map(e=>{
      const when = new Date(e.timestamp).toLocaleString();
      const safe = (e.content||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      return `<div class="item"><div>${safe}</div><div class="small">${when}</div></div>`;
    }).join('') || '<div class="small">No entries yet.</div>';
  }catch(e){
    el('journalStatus').innerText = 'Failed to load: '+(e.message||e);
  }
}

async function saveJournal(){
  const txt = (el('journalText').value||'').trim();
  const consent = el('journalConsent').checked;
  if(!txt){ el('journalStatus').innerText='Write something first.'; return; }
  if(!consent){ el('journalStatus').innerText='Please check the consent box.'; return; }
  el('journalStatus').innerText='Saving...';
  try{
    await req('/journals/add','POST',{content: txt, consent});
    el('journalText').value=''; el('journalConsent').checked=false;
    el('journalStatus').innerText='Saved!';
    refreshJournal();
  }catch(e){
    el('journalStatus').innerText='Failed: '+(e.message||e);
  }
}

document.addEventListener('DOMContentLoaded', ()=>{
  // Server key persistence
  el('gatewayKey').value = localStorage.getItem('gw') || '';
  API_KEY = el('gatewayKey').value;
  el('gatewayKey').addEventListener('change', ()=>{ API_KEY = el('gatewayKey').value; localStorage.setItem('gw', API_KEY) });

  // Login
  el('loginBtn').onclick = async ()=>{
    API_KEY = el('gatewayKey').value;
    const username = el('loginUsername').value.trim();
    const password = el('loginPassword').value.trim();
    const res = await req('/auth/login','POST',{username,password});
    SESSION = res.token; ME = res.user;
    setAuthVisible(true); renderMe(); connectWS(); await refreshUsers(); await refreshJournal();
  };

  // Signup
  el('signupBtn').onclick = async ()=>{
    API_KEY = el('gatewayKey').value;
    const username = el('signupUsername').value.trim();
    const display_name = el('signupDisplayName').value.trim() || username;
    const password = el('signupPassword').value.trim();
    if(!username || !password){ alert('Username and password required'); return; }
    await req('/auth/signup','POST',{username,display_name,password});
    alert('Signed up! Now login using the Login form above.');
  };

  // Logout
  el('logoutBtn').onclick = async ()=>{
    await req('/auth/logout','POST',{}); SESSION=''; ME=null; setAuthVisible(false); location.reload();
  };

  // AI toggle
  el('aiToggle').onchange = async (e)=>{
    await req(`/users/me/ai-toggle?enabled=${e.target.checked}`,'PATCH');
    ME.ai_enabled = e.target.checked;
  };

  // Relationship save
  el('relSave').onclick = async ()=>{
    const other = el('relOther').value;
    const relation = el('relKind').value;
    if(!other) return;
    await req('/relationships/set','POST', {other_user_id: other, relation});
    el('relStatus').innerText = 'Saved!';
    setTimeout(()=>{el('relStatus').innerText=''},1500);
    await refreshUsers();
  };

  // Journal buttons
  el('saveJournalBtn').onclick = saveJournal;
  el('refreshJournalBtn').onclick = refreshJournal;

  setAuthVisible(false);
});
</script>
</body>
</html>
    """
    return HTMLResponse(html)

# ---------------------
# Auth routes
# ---------------------
from typing import Optional as _Optional

def require_api_key(x_api_key: _Optional[str] = Header(None)):
    expected = (PUBLIC_UI_API_KEY or "").strip()
    # If PUBLIC_UI_API_KEY empty or "disabled", skip the check
    if expected and expected.lower() != "disabled":
        if x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/auth/signup")
async def signup(req: SignupRequest, _: None = Depends(require_api_key)):
    await get_mongo_client()
    existing = await users_col.find_one({"username": req.username})
    if existing:
        raise HTTPException(status_code=400, detail="Username taken")
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    h = hash_password(req.password)
    now = datetime.now(pytz.UTC)
    doc = {
        "user_id": user_id,
        "username": req.username,
        "display_name": req.display_name,
        "password_salt": h["salt"],
        "password_hash": h["hash"],
        "ai_enabled": False,
        "created_at": now,
        "last_seen": now
    }
    await users_col.insert_one(doc)
    return {"ok": True, "user_id": user_id}

@app.post("/auth/login")
async def login(req: LoginRequest, _: None = Depends(require_api_key)):
    await get_mongo_client()
    user = await users_col.find_one({"username": req.username})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(req.password, user.get("password_salt",""), user.get("password_hash","")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = await create_session(user["user_id"])
    await users_col.update_one({"user_id": user["user_id"]}, {"$set": {"last_seen": datetime.now(pytz.UTC)}})
    return {"token": token, "user": {"user_id": user["user_id"], "username": user["username"], "display_name": user["display_name"], "ai_enabled": user.get("ai_enabled", False)}}

@app.post("/auth/logout")
async def logout(sess=Depends(require_session), _: None = Depends(require_api_key)):
    await sessions_col.delete_one({"token": sess["token"]})
    return {"ok": True}

# ---------------------
# Users / Relationships / Presence
# ---------------------
@app.get("/users/me")
async def users_me(sess=Depends(require_session), _: None = Depends(require_api_key)):
    u = sess["user"]
    return {"user": {"user_id": u["user_id"], "username": u["username"], "display_name": u["display_name"], "ai_enabled": u.get("ai_enabled", False)}}

@app.patch("/users/me/ai-toggle")
async def toggle_ai(enabled: bool = Query(...), sess=Depends(require_session), _: None = Depends(require_api_key)):
    await users_col.update_one({"user_id": sess["user"]["user_id"]}, {"$set": {"ai_enabled": bool(enabled)}})
    return {"ok": True, "ai_enabled": bool(enabled)}

@app.get("/users/list")
async def users_list(sess=Depends(require_session), _: None = Depends(require_api_key)):
    me_id = sess["user"]["user_id"]
    all_users = await users_col.find({}, {"password_hash": 0, "password_salt": 0}).to_list(length=None)
    rels = { r["other_user_id"]: r["relation"] async for r in relationships_col.find({"user_id": me_id}) }
    async with manager.lock:
        online = set(manager.active.keys())
    users = []
    for u in all_users:
        users.append({
            "user_id": u["user_id"],
            "username": u["username"],
            "display_name": u["display_name"],
            "online": u["user_id"] in online,
            "relation": rels.get(u["user_id"]),
            "ai_enabled": bool(u.get("ai_enabled", False))
        })
    return {"users": users}

# --- Relationship utils (NEW) ---
INVERSE_REL = {
    "mother": "son",
    "father": "son",
    "son": "father",
    "daughter": "mother",
    "sister": "brother",
    "brother": "sister",
    "wife": "husband",
    "husband": "wife",
    "friend": "friend"
}

async def resolve_target_role_for_reply(speaker_id: str, target_id: str) -> str:
    """
    Determine the role of the target relative to the speaker, from target's perspective.
    The target is the one generating the reply, so we need how *target* views *speaker*.
    """
    await get_mongo_client()
    doc = await relationships_col.find_one({"user_id": target_id, "other_user_id": speaker_id})
    role = (doc or {}).get("relation", "").strip().lower()
    return role if role else "friend"

@app.post("/relationships/set")
async def rel_set(req: RelationshipSetRequest, sess=Depends(require_session), _: None = Depends(require_api_key)):
    me_id = sess["user"]["user_id"]
    now = datetime.now(pytz.UTC)
    rel = req.relation.strip().lower()

    # forward: me -> other
    await relationships_col.update_one(
        {"user_id": me_id, "other_user_id": req.other_user_id},
        {"$set": {"relation": rel, "updated_at": now}},
        upsert=True
    )
    # inverse: other -> me
    inv = INVERSE_REL.get(rel, "friend")
    await relationships_col.update_one(
        {"user_id": req.other_user_id, "other_user_id": me_id},
        {"$set": {"relation": inv, "updated_at": now}},
        upsert=True
    )
    return {"ok": True}

@app.get("/relationships/with/{other_id}")
async def rel_get(other_id: str, sess=Depends(require_session), _: None = Depends(require_api_key)):
    me_id = sess["user"]["user_id"]
    r = await relationships_col.find_one({"user_id": me_id, "other_user_id": other_id})
    return {"relation": (r or {}).get("relation")}

# ---------------------
# Core AI pieces
# ---------------------
def preprocess_input(user_input: str) -> str:
    try:
        doc = nlp(user_input)
        key_terms = []
        for t in doc:
            if hasattr(t, "pos_") and hasattr(t, "is_stop"):
                if t.pos_ in ["NOUN", "VERB"] and not t.is_stop:
                    key_terms.append(t.text.lower())
            else:
                key_terms.append(t.text.lower())
        extra_terms = []
        for term in key_terms:
            try:
                syns = wordnet.synsets(term)
            except Exception:
                syns = []
            synonyms = set()
            for syn in syns:
                for lemma in syn.lemmas():
                    w = lemma.name().replace('_',' ')
                    if w != term and len(w.split()) <= 2:
                        synonyms.add(w)
            extra_terms.extend(list(synonyms)[:3])
        if extra_terms:
            user_input += " " + " ".join(set(extra_terms[:10]))
        return user_input
    except Exception:
        return user_input

async def get_recent_conversation_history(speaker_id: str, target_id: str, limit: int = 6) -> List[dict]:
    await get_mongo_client()
    pipeline = [
        {"$match": {
            "user_id": {"$all": [speaker_id, target_id]},
            "$or": [{"speaker_id": speaker_id, "target_id": target_id},
                    {"speaker_id": target_id, "target_id": speaker_id}]
        }},
        {"$sort": {"timestamp": -1}},
        {"$limit": limit},
        {"$sort": {"timestamp": 1}}
    ]
    out = []
    async for conv in conversations_col.aggregate(pipeline):
        sp_name = conv.get("speaker_name")
        if not sp_name:
            u = await users_col.find_one({"user_id": conv["speaker_id"]})
            sp_name = (u or {}).get("display_name") or (u or {}).get("username") or conv["speaker_id"]
        raw_ts = as_utc_aware(conv["timestamp"])
        out.append({
            "speaker": sp_name,
            "content": conv["content"],
            "timestamp": raw_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "type": conv.get("type","user_input"),
            "source": conv.get("source", "human"),
            "raw_timestamp": raw_ts,
            "conversation_id": conv["conversation_id"]
        })
    return out

async def generate_personality_traits(user_id: str) -> dict:
    await get_mongo_client()
    convs = [doc async for doc in conversations_col.find({"user_id": user_id}).sort("timestamp", -1).limit(500)]
    journals = [doc async for doc in journals_col.find({"user_id": user_id}).sort("timestamp", -1).limit(500)]
    data_text = "\n".join([c.get("content","") for c in convs] + [j.get("content","") for j in journals])[:1000]
    if not data_text:
        return {"core_traits": {}, "sub_traits": []}
    cached = await personalities_col.find_one({"user_id": user_id})
    if cached and "traits" in cached:
        return cached["traits"]

    u = await users_col.find_one({"user_id": user_id})
    big_five_prompt = f"""
    Analyze this text from {(u or {}).get('display_name', user_id)}:
    {data_text}
    Return a JSON object with:
    - "core_traits": 5 traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) with scores (0-100) and one-sentence explanations.
    - "sub_traits": 3 unique traits with one-sentence descriptions.
    Ensure the response is concise to fit within 700 tokens.
    """
    traits = None
    for attempt in range(3):
        try:
            resp = await (await get_openai_client()).chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role":"system","content":"You are a helpful assistant that generates personality traits."},
                    {"role":"user","content":big_five_prompt}
                ],
                max_tokens=700, temperature=0.7
            )
            txt = resp.choices[0].message.content.strip()
            txt = re.sub(r'^```json\s*|\s*```$', '', txt, flags=re.MULTILINE).strip()
            traits = json.loads(txt)
            if "core_traits" in traits and "sub_traits" in traits:
                if isinstance(traits["core_traits"], list):
                    traits["core_traits"] = {t["trait"]: {"score": t["score"], "explanation": t["explanation"]} for t in traits["core_traits"]}
                break
        except Exception:
            if attempt == 2:
                traits = {
                    "core_traits": {
                        "Openness":{"score":50,"explanation":"Neutral openness."},
                        "Conscientiousness":{"score":50,"explanation":"Neutral conscientiousness."},
                        "Extraversion":{"score":50,"explanation":"Neutral extraversion."},
                        "Agreeableness":{"score":50,"explanation":"Neutral agreeableness."},
                        "Neuroticism":{"score":50,"explanation":"Neutral neuroticism."}
                    },
                    "sub_traits":[
                        {"trait":"neutral","description":"Shows balanced behavior."},
                        {"trait":"adaptable","description":"Adapts to context."},
                        {"trait":"curious","description":"Engages with data."}
                    ]
                }
    await personalities_col.update_one({"user_id":user_id},{"$set":{"traits":traits}}, upsert=True)
    return traits

async def get_greeting_and_tone(bot_role: str, target_id: str) -> Tuple[str,str]:
    await get_mongo_client()
    key = f"greeting_{target_id}_{bot_role}"
    cached = await greetings_cache_col.find_one({"key": key, "timestamp": {"$gte": datetime.now(pytz.UTC)-timedelta(hours=1)}})
    if cached:
        return cached["greeting"], cached["tone"]

    saved = await saved_greetings_col.find_one({"target_id": target_id, "bot_role": bot_role.lower()}, sort=[("timestamp",-1)])
    if saved:
        return saved["greeting"], "warm, youthful" if bot_role.lower() in ["daughter","son"] else "nurturing, caring"

    defaults = {
        "daughter": ("Hey, Mom", "warm, youthful"),
        "son": ("Hey, Mom", "warm, youthful"),
        "mother": ("Hi, sweetie", "nurturing, caring"),
        "father": ("Hey, kid", "warm, supportive"),
        "sister": ("Yo, sis", "playful, casual"),
        "brother": ("Yo, bro", "playful, casual"),
        "wife": ("Hey, hon", "affectionate, conversational"),
        "husband": ("Hey, hon", "affectionate, conversational"),
        "friend": ("Hey, what's good?", "casual, friendly")
    }
    greeting, tone = defaults.get(bot_role.lower(), ("Hey","casual, friendly"))

    traits = await generate_personality_traits(target_id)
    prompt = f"""
    You are generating a greeting for a {bot_role} with traits: {', '.join(traits.get('core_traits', {}).keys())}.
    Return a JSON object: {{"greeting":"short greeting","tone":"tone description"}}
    """
    for attempt in range(3):
        try:
            resp = await (await get_openai_client()).chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role":"system","content":"Return only valid JSON with 'greeting' and 'tone' keys."},
                    {"role":"user","content":prompt}
                ], max_tokens=100, temperature=0.5
            )
            txt = resp.choices[0].message.content.strip()
            txt = re.sub(r'^```json\s*|\s*```$','',txt, flags=re.MULTILINE).strip()
            obj = json.loads(txt)
            if "greeting" in obj and "tone" in obj:
                greeting, tone = obj["greeting"], obj["tone"]
                break
        except Exception:
            if attempt==2: break

    await greetings_cache_col.update_one({"key":key},{"$set":{"greeting":greeting,"tone":tone,"timestamp":datetime.now(pytz.UTC)}}, upsert=True)
    return greeting, tone

# ---------------------
# RAG: memories (convos + journals)
# ---------------------
async def find_relevant_memories(speaker_id: str, user_id: str, user_input: str, speaker_name: str, max_memories: int = 5) -> List[dict]:
    global faiss_store
    await ensure_faiss_store()
    await get_mongo_client()
    loop = asyncio.get_event_loop()
    processed = await loop.run_in_executor(None, preprocess_input, user_input)
    cache_key = f"input_{hash(processed)}"
    if cache_key in embedding_cache:
        _ = embedding_cache[cache_key]
    else:
        _ = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
        embedding_cache[cache_key] = _

    udoc = await users_col.find_one({"user_id": user_id})
    target_name = (udoc or {}).get("display_name") or (udoc or {}).get("username") or user_id

    results = await loop.run_in_executor(None, lambda: faiss_store.similarity_search_with_score(processed, k=max_memories*3))
    mems = []
    for doc, score in results:
        md = doc.metadata
        item_id = md.get("item_id"); item_type = md.get("item_type")
        if not item_id or not item_type: continue
        col = conversations_col if item_type=="conversation" else journals_col
        id_field = "conversation_id" if item_type=="conversation" else "entry_id"
        base = await col.find_one({id_field: item_id})
        if not base:
            continue
        
        uids = base.get("user_id", [])
        if isinstance(uids, list):
            if user_id not in uids:
                continue
        else:
            if uids != user_id:
                continue

        if item_type=="journal":
            base["speaker_name"] = target_name

        adjusted = 1.0 - score
        if item_type=="journal": adjusted += 0.9
        elif md.get("speaker_id")==speaker_id or md.get("target_id")==user_id: adjusted += 0.7
        if speaker_name.lower() in base.get("content","").lower() or target_name.lower() in base.get("content","").lower():
            adjusted += 0.3
        ts = as_utc_aware(md.get("timestamp")) or as_utc_aware(base.get("timestamp"))
        days_old = (datetime.now(pytz.UTC) - ts).days if ts else 9999
        temporal_weight = 1/(1 + np.log1p(max(days_old,1)/30))
        adjusted *= temporal_weight
        if adjusted < 0.3:
            logger.debug(f"skip memory {item_id}: low adjusted={adjusted:.3f} type={item_type}")
            continue
        mems.append({
            "type": item_type, "content": base["content"], "timestamp": as_utc_aware(base["timestamp"]),
            "score": float(adjusted), "user_id": md.get("user_id", []),
            "speaker_id": md.get("speaker_id"), "speaker_name": base.get("speaker_name", target_name),
            "target_id": md.get("target_id"), "target_name": md.get("target_name")
        })
    mems.sort(key=lambda x: x["score"], reverse=True)
    return mems[:max_memories]

async def should_include_memories(user_input: str, speaker_id: str, user_id: str) -> Tuple[bool, List[dict]]:
    sp = await users_col.find_one({"user_id": speaker_id})
    speaker_name = (sp or {}).get("display_name") or (sp or {}).get("username") or speaker_id
    mems = await find_relevant_memories(speaker_id, user_id, user_input, speaker_name, max_memories=10)
    if not mems: return False, []
    loop = asyncio.get_event_loop()
    processed = await loop.run_in_executor(None, preprocess_input, user_input)
    inp = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
    rel = []
    for m in mems:
        emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(m["content"]))
        sim = np.dot(inp, emb) / (np.linalg.norm(inp)*np.linalg.norm(emb))
        threshold = 0.35 if m["type"] == "journal" else 0.45
        if sim >= threshold:
            rel.append(m)
    return (len(rel)>0), rel[:3]

# ---------------------
# initialize_bot (role auto-detect)
# ---------------------
async def initialize_bot(speaker_id: str, target_id: str, bot_role: Optional[str], user_input: str) -> Tuple[str,str,bool]:
    sp = await users_col.find_one({"user_id": speaker_id})
    tg = await users_col.find_one({"user_id": target_id})
    if not sp or not tg:
        raise ValueError("Invalid IDs")

    # auto-resolve role if not provided or unhelpful
    role_in = (bot_role or "").strip().lower()
    if not role_in or role_in == "friend":
        role_in = await resolve_target_role_for_reply(speaker_id, target_id)

    traits = await generate_personality_traits(target_id)
    recent = await get_recent_conversation_history(speaker_id, target_id)

    history_for_prompt = recent[:]
    if recent:
        last = recent[-1]
        if last.get("content","").strip() == user_input.strip():
            history_for_prompt = recent[:-1]

    allow_repeat_ref = False
    try:
        loop = asyncio.get_event_loop()
        q_emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(user_input))
        for m in history_for_prompt[-10:]:
            if not m.get("content"): continue
            emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(m["content"]))
            sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb)))
            if sim >= 0.92:
                allow_repeat_ref = True
                break
    except Exception:
        allow_repeat_ref = False

    if history_for_prompt:
        hist_text = "\n".join([f"[{m['raw_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {m['content']}" for m in history_for_prompt])
        last_ts = history_for_prompt[-1]["raw_timestamp"]
    else:
        hist_text = "No earlier messages."
        last_ts = None

    use_greeting = (not history_for_prompt) or (datetime.now(pytz.UTC)-as_utc_aware(last_ts)).total_seconds()/60 > 30
    greeting, tone = await get_greeting_and_tone(role_in, target_id)

    include, mems = await should_include_memories(user_input, speaker_id, target_id)
    mems_text = "No relevant memories."
    if include and mems:
        good = [m for m in mems if all(k in m for k in ["content","type","timestamp","speaker_name"])]
        if good:
            # Present compact, clearly attributed memories; journals show as “(journal, you)”
            def who(m):
                return "you" if (m["type"] == "journal") else m["speaker_name"]
            mems_text = "\n".join([
                f"- {m['content']} ({m['type']}, {m['timestamp'].strftime('%Y-%m-%d')}, said by {who(m)})"
                for m in good
            ])

    rails = f"""
    Grounding rules:
    - You may reference dates/timestamps in the earlier conversation history.
    - Do NOT refer to the current message as if it were a past event.
    - If multiple memories conflict, **prioritize the TARGET user's own journal entries** over conversations or others’ journals.
    - If a preference is in the TARGET user's journal (e.g., food likes/dislikes), treat it as the source of truth unless the TARGET explicitly overrides it in the *current* message.
    - Only say "you asked this before..." if there is a clearly earlier, highly similar message. Permission: {"ALLOWED" if allow_repeat_ref else "NOT ALLOWED"}.
    - If NOT ALLOWED, avoid implying repetition; respond normally.
    """


    trait_str = ', '.join([f"{k} ({v['explanation']})" for k,v in list(traits.get('core_traits', {}).items())[:3]]) or "balanced"
    sp_name = (sp or {}).get("display_name") or (sp or {}).get("username") or speaker_id
    tg_name = (tg or {}).get("display_name") or (tg or {}).get("username") or target_id

    base_prompt = f"""
    You are {tg_name}, responding as an AI Twin to {sp_name}, their {role_in}.
    Use a {tone} tone and reflect your personality: {trait_str}.

    Earlier conversation (timestamps included, excludes the current message):
    {hist_text}

    {rails}

- {'Start with "' + greeting + '" if no earlier messages or time gap > 30 minutes.' if use_greeting else 'Do not start with a greeting.'}
- Keep it short (2–3 sentences), natural, and personalized.
- If relevant to the current input, **weave in up to 1–2 of these memories naturally**, clearly attributing them (e.g., “I wrote in my journal…”, “you said…”):
{mems_text}
- **Prioritize the TARGET’s own journal** for facts about the TARGET; do not contradict it unless the TARGET explicitly changes that fact in the current message.
Current user input: {user_input}

Respond directly to the Current user input above.

    """

    if include:
        base_prompt = base_prompt.replace("{rails}\n\n", "{rails}\n\nPotentially relevant memories:\n" + mems_text + "\n\n")

    return base_prompt, greeting, use_greeting

async def generate_response(prompt: str, user_input: str, greeting: str, use_greeting: bool) -> str:
    try:
        resp = await (await get_openai_client()).chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are an AI Twin responding in a personalized, casual manner."},
                {"role":"user","content":prompt}
            ],
            max_tokens=200, temperature=0.6
        )
        text = resp.choices[0].message.content.strip()
        if len(text.split()) >= 4 and ((use_greeting and text.lower().startswith(greeting.lower())) or not use_greeting):
            parts = text.split('. ')[:3]
            text = '. '.join([p for p in parts if p]).strip()
            if text and not text.endswith('.'): text += '.'
            return text
    except Exception as e:
        await errors_col.insert_one({"error": str(e), "input": user_input, "timestamp": datetime.now(pytz.UTC)})
    return f"{greeting}, sounds cool! What's up?" if use_greeting else "Sounds cool! What's up?"

# ---------------------
# Save message helper
# ---------------------
async def save_and_embed_message(speaker_id: str, target_id: str, text: str, source: str) -> dict:
    await get_mongo_client()
    await ensure_faiss_store()
    sp = await users_col.find_one({"user_id": speaker_id})
    tg = await users_col.find_one({"user_id": target_id})
    sp_name = (sp or {}).get("display_name") or (sp or {}).get("username") or speaker_id
    tg_name = (tg or {}).get("display_name") or (tg or {}).get("username") or target_id
    now = datetime.now(pytz.UTC)
    conv_id = str(uuid.uuid4())
    doc = {
        "conversation_id": conv_id,
        "user_id": [speaker_id, target_id],
        "speaker_id": speaker_id,
        "speaker_name": sp_name,
        "target_id": target_id,
        "target_name": tg_name,
        "content": text,
        "type": "user_input" if source=="human" else "response",
        "source": source,
        "timestamp": now
    }
    await conversations_col.insert_one(doc)

    processed = preprocess_input(text)
    loop = asyncio.get_event_loop()
    emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
    await embeddings_col.insert_one({
        "item_id": conv_id, "item_type":"conversation", "user_id":[speaker_id,target_id],
        "speaker_id": speaker_id, "speaker_name": sp_name,
        "target_id": target_id, "target_name": tg_name,
        "embedding": emb, "timestamp": now, "content": text
    })
    try:
        db_doc = Document(page_content=text, metadata={
            "item_id": conv_id, "item_type":"conversation", "user_id":[speaker_id,target_id],
            "speaker_id": speaker_id, "speaker_name": sp_name,
            "target_id": target_id, "target_name": tg_name,
            "timestamp": now
        })
        with faiss_lock:
            faiss_store.add_documents([db_doc])
            faiss_store.save_local(FAISS_DIR)
    except Exception as e:
        logger.warning(f"FAISS add fail: {e}")

    return doc

# ---------------------
# HTTP Chat
# ---------------------
def require_api_and_session(sess=Depends(require_session), _: None = Depends(require_api_key)):
    return sess

@app.post("/send_message", response_model=MessageResponse)
async def send_message(req: MessageRequest, sess=Depends(require_api_and_session)):
    if sess["user"]["user_id"] != req.speaker_id:
        raise HTTPException(status_code=403, detail="Sender mismatch")
    await save_and_embed_message(req.speaker_id, req.target_id, req.user_input, source="human")
    tg = await users_col.find_one({"user_id": req.target_id})
    if tg and tg.get("ai_enabled", False):
        # Let server resolve role if not helpful
        prompt, greeting, use_greeting = await initialize_bot(req.speaker_id, req.target_id, getattr(req, "bot_role", None), req.user_input)
        ai_text = await generate_response(prompt, req.user_input, greeting, use_greeting)
        await save_and_embed_message(req.target_id, req.speaker_id, ai_text, source="ai_twin")
        return MessageResponse(response=ai_text)
    return MessageResponse(response="Sent.")

@app.get("/conversations/with/{other_id}")
async def history_with(other_id: str, limit: int = 30, sess=Depends(require_api_and_session)):
    me = sess["user"]["user_id"]
    cur = conversations_col.find({"user_id": {"$all":[me, other_id]}}).sort("timestamp",-1).limit(limit)
    out = []
    async for c in cur:
        out.append({
            "conversation_id": c["conversation_id"],
            "speaker_id": c["speaker_id"],
            "target_id": c["target_id"],
            "content": c["content"],
            "source": c.get("source","human"),
            "timestamp": as_utc_aware(c["timestamp"]).isoformat()
        })
    return {"messages": list(reversed(out))}

# ---------------------
# Journal endpoints
# ---------------------
@app.post("/journals/add")
async def journals_add(req: JournalAddRequest, sess=Depends(require_api_and_session)):
    if not req.consent:
        raise HTTPException(status_code=400, detail="Consent required: please confirm the checkbox.")
    await get_mongo_client()
    now = datetime.now(pytz.UTC)
    entry_id = str(uuid.uuid4())
    doc = {
        "entry_id": entry_id,
        "user_id": [sess["user"]["user_id"]],
        "content": (req.content or "").strip(),
        "timestamp": now
    }
    await journals_col.insert_one(doc)
    try:
        await process_new_entry(item_id=entry_id, item_type="journal", content=doc["content"], user_id=doc["user_id"])
    except Exception:
        pass
    return {"ok": True, "entry_id": entry_id, "timestamp": now.isoformat()}

@app.get("/journals/list")
async def journals_list(limit: int = 20, sess=Depends(require_api_and_session)):
    me = sess["user"]["user_id"]
    cur = journals_col.find({"user_id": {"$in": [me]}}).sort("timestamp", -1).limit(limit)
    out = []
    async for j in cur:
        out.append({
            "entry_id": j["entry_id"],
            "content": j.get("content",""),
            "timestamp": as_utc_aware(j.get("timestamp")).isoformat() if j.get("timestamp") else None
        })
    return {"entries": out}

# ---------------------
# WebSocket Chat
# ---------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    user_id = websocket.query_params.get("user_id")
    await get_mongo_client()
    sess = await sessions_col.find_one({"token": token, "user_id": user_id})
    if not sess:
        await websocket.close(code=4401)
        return

    try:
        await manager.connect(user_id, websocket)
        await manager.broadcast_presence()
        await users_col.update_one({"user_id": user_id}, {"$set":{"last_seen": datetime.now(pytz.UTC)}})
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except Exception:
                continue
            if msg.get("type") == "chat":
                to = msg["to"]
                text = msg["text"]
                saved = await save_and_embed_message(user_id, to, text, source="human")
                await manager.send_to(to, {"type":"chat","from": user_id, "payload":{
                    "speaker_id": saved["speaker_id"],
                    "target_id": saved["target_id"],
                    "content": saved["content"],
                    "source": "human",
                    "timestamp": saved["timestamp"].isoformat()
                }})
                tgt = await users_col.find_one({"user_id": to})
                if tgt and tgt.get("ai_enabled", False):
                    # auto-role resolve
                    prompt, greeting, use_greeting = await initialize_bot(user_id, to, None, text)
                    ai_text = await generate_response(prompt, text, greeting, use_greeting)
                    ai_saved = await save_and_embed_message(to, user_id, ai_text, source="ai_twin")
                    await manager.send_to(user_id, {"type":"ai","from": to, "payload":{
                        "speaker_id": ai_saved["speaker_id"],
                        "target_id": ai_saved["target_id"],
                        "content": ai_saved["content"],
                        "source": "ai_twin",
                        "timestamp": ai_saved["timestamp"].isoformat()
                    }})
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(user_id)
        await manager.broadcast_presence()

# ---------------------
# Change streams
# ---------------------
async def process_new_entry(item_id: str, item_type: str, content: str, user_id: list,
                            speaker_id: Optional[str] = None, speaker_name: Optional[str] = None,
                            target_id: Optional[str] = None, target_name: Optional[str] = None):
    try:
        await get_mongo_client()
        await ensure_faiss_store()
        processed = preprocess_input(content)
        loop = asyncio.get_event_loop()
        emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
        now = datetime.now(pytz.UTC)
        doc = {
            "item_id": item_id, "item_type": item_type, "user_id": user_id,
            "content": content, "embedding": emb, "timestamp": now
        }
        if item_type=="conversation":
            doc.update({"speaker_id": speaker_id, "speaker_name": speaker_name, "target_id": target_id, "target_name": target_name})
        await embeddings_col.insert_one(doc)

        with faiss_lock:
            if faiss_store is None:
                faiss_store = FAISS.from_texts(["empty"], embeddings)
            meta = {"item_id": item_id, "item_type": item_type, "user_id": user_id, "timestamp": now}
            if item_type=="conversation":
                meta.update({"speaker_id": speaker_id, "speaker_name": speaker_name, "target_id": target_id, "target_name": target_name})
            faiss_store.add_documents([Document(page_content=content, metadata=meta)])
            faiss_store.save_local(FAISS_DIR)
    except Exception as e:
        await errors_col.insert_one({"error": str(e), "item_id": item_id, "item_type": item_type, "timestamp": datetime.now(pytz.UTC)})

async def watch_conversations():
    while True:
        try:
            await get_mongo_client()
            async with conversations_col.watch([{"$match":{"operationType":"insert"}}], full_document="updateLookup") as stream:
                async for change in stream:
                    doc = change["fullDocument"]
                    if doc.get("type") == "user_input" and doc.get("source") == "human":
                        await process_new_entry(
                            item_id=doc["conversation_id"], item_type="conversation",
                            content=doc["content"], user_id=doc["user_id"],
                            speaker_id=doc.get("speaker_id"), speaker_name=doc.get("speaker_name"),
                            target_id=doc.get("target_id"), target_name=doc.get("target_name")
                        )
        except Exception:
            await errors_col.insert_one({"error": "watch_conversations error", "timestamp": datetime.now(pytz.UTC)})
            await asyncio.sleep(5)

async def watch_journals():
    while True:
        try:
            await get_mongo_client()
            async with journals_col.watch([{"$match":{"operationType":"insert"}}], full_document="updateLookup") as stream:
                async for change in stream:
                    doc = change["fullDocument"]
                    await process_new_entry(item_id=doc["entry_id"], item_type="journal", content=doc["content"], user_id=doc["user_id"])
        except Exception:
            await errors_col.insert_one({"error": "watch_journals error", "timestamp": datetime.now(pytz.UTC)})
            await asyncio.sleep(5)

async def watch_collections():
    await asyncio.gather(watch_conversations(), watch_journals())

# ---------------------
# Demo seed / initialization
# ---------------------
async def clear_database():
    await get_mongo_client()
    await users_col.delete_many({})
    await conversations_col.delete_many({})
    await journals_col.delete_many({})
    await embeddings_col.delete_many({})
    await relationships_col.delete_many({})
    await sessions_col.delete_many({})

async def populate_users():
    now = datetime.now(pytz.UTC)
    def mkuser(uid, uname, name):
        h=hash_password("password")
        return {"user_id": uid, "username": uname, "display_name": name, "password_salt":h["salt"], "password_hash":h["hash"], "ai_enabled": False, "created_at": now, "last_seen": now}
    base = [
        mkuser("user1","nipa","Nipa"),
        mkuser("user2","nick","Nick"),
        mkuser("user3","arif","Arif"),
        mkuser("user4","diana","Diana")
    ]
    for u in base:
        if not await users_col.find_one({"user_id": u["user_id"]}):
            await users_col.insert_one(u)

async def batch_embed_texts(texts: List[str]):
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: embeddings.embed_documents(texts))
    except Exception:
        return [None]*len(texts)

async def populate_conversations():
    now = datetime.now(pytz.UTC)
    convs = [
        {"conversation_id": str(uuid.uuid4()), "user_id":["user1","user2"], "speaker_id":"user1","speaker_name":"Nipa","target_id":"user2","target_name":"Nick","content":"Hey nick, ready for the project?","type":"user_input","source":"human","timestamp": now - timedelta(days=1)},
        {"conversation_id": str(uuid.uuid4()), "user_id":["user2","user1"], "speaker_id":"user2","speaker_name":"Nick","target_id":"user1","target_name":"Nipa","content":"Yeah, let's do this!","type":"user_input","source":"human","timestamp": now - timedelta(days=1, hours=1)},
        {"conversation_id": str(uuid.uuid4()), "user_id":["user3","user4"], "speaker_id":"user3","speaker_name":"Arif","target_id":"user4","target_name":"Diana","content":"Diana, got any weekend plans?","type":"user_input","source":"human","timestamp": now - timedelta(days=2)},
        {"conversation_id": str(uuid.uuid4()), "user_id":["user4","user3"], "speaker_id":"user4","speaker_name":"Diana","target_id":"user3","target_name":"Arif","content":"Just chilling, you?","type":"user_input","source":"human","timestamp": now - timedelta(days=2, hours=1)},
        {"conversation_id": str(uuid.uuid4()), "user_id":["user1","user3"], "speaker_id":"user1","speaker_name":"Nipa","target_id":"user3","target_name":"Arif","content":"Dad, I want to go to disney","type":"user_input","source":"human","timestamp": now - timedelta(hours=12)},
        {"conversation_id": str(uuid.uuid4()), "user_id":["user4","user2"], "speaker_id":"user4","speaker_name":"Diana","target_id":"user2","target_name":"Nick","content":"Nick, have you tried the new coffee shop yet?","type":"user_input","source":"human","timestamp": now - timedelta(hours=10)}
    ]
    for c in convs:
        if not await conversations_col.find_one({"conversation_id": c["conversation_id"]}):
            await conversations_col.insert_one(c)
    embeddings_result = await batch_embed_texts([c["content"] for c in convs])
    docs = []
    for c, e in zip(convs, embeddings_result):
        if e is not None and not await embeddings_col.find_one({"item_id": c["conversation_id"], "item_type":"conversation"}):
            docs.append({
                "item_id": c["conversation_id"], "item_type":"conversation", "user_id": c["user_id"],
                "content": c["content"], "embedding": e, "timestamp": c["timestamp"],
                "speaker_id": c["speaker_id"], "speaker_name": c["speaker_name"],
                "target_id": c["target_id"], "target_name": c["target_name"]
            })
    if docs: await embeddings_col.insert_many(docs)

async def populate_journals():
    now = datetime.now(pytz.UTC)
    j = {"entry_id": str(uuid.uuid4()), "user_id": ["user1"], "content":"I am in love with Jack", "timestamp": now - timedelta(hours=6)}
    if not await journals_col.find_one({"entry_id": j["entry_id"]}):
        await journals_col.insert_one(j)
    emb = (await batch_embed_texts([j["content"]]))[0]
    if emb is not None and not await embeddings_col.find_one({"item_id": j["entry_id"], "item_type":"journal"}):
        await embeddings_col.insert_one({"item_id": j["entry_id"], "item_type":"journal", "user_id": j["user_id"], "content": j["content"], "embedding": emb, "timestamp": j["timestamp"]})

async def verify_data():
    counts = {
        "Users": await users_col.count_documents({}),
        "Conversations": await conversations_col.count_documents({}),
        "Journals": await journals_col.count_documents({}),
        "Embeddings": await embeddings_col.count_documents({})
    }
    logger.info(f"DB counts: {counts}")

async def initialize_db():
    if SEED_DEMO:
        await clear_database()
        await populate_users()
        await populate_conversations()
        await populate_journals()
        await verify_data()
    await initialize_faiss_store()

# ---------------------
# Run (local only)
# ---------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, proxy_headers=True, timeout_keep_alive=70)




# import os
# import json
# import re
# import hashlib
# import secrets
# import base64
# import uuid
# import threading
# import asyncio
# import logging
# from datetime import datetime, timedelta
# from typing import List, Optional, Tuple, Dict, Any
# from contextlib import asynccontextmanager

# import pytz
# import numpy as np
# from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, Depends, Query
# from fastapi.responses import HTMLResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from motor.motor_asyncio import AsyncIOMotorClient
# from dotenv import load_dotenv

# # LangChain / FAISS / OpenAI
# from cachetools import TTLCache
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.docstore.document import Document
# from openai import AsyncOpenAI

# # ---------------------
# # Setup logging
# # ---------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("ai-twin-app")

# # ---------------------
# # NLTK & spaCy (hardened for Render)
# # ---------------------
# NLTK_DATA_DIR = os.getenv("NLTK_DATA", "/tmp/nltk_data")
# os.environ["NLTK_DATA"] = NLTK_DATA_DIR
# os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# import nltk
# nltk.download('wordnet', quiet=True, download_dir=NLTK_DATA_DIR)
# nltk.download('punkt', quiet=True, download_dir=NLTK_DATA_DIR)
# from nltk.corpus import wordnet

# import spacy
# try:
#     nlp = spacy.load("en_core_web_sm")
# except Exception:
#     logger.warning("spaCy model 'en_core_web_sm' not available; using spacy.blank('en') fallback.")
#     nlp = spacy.blank("en")

# # ---------------------
# # Env & constants
# # ---------------------
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# MONGODB_URI = os.getenv("MONGODB_URI")
# PUBLIC_UI_API_KEY = os.getenv("PUBLIC_UI_API_KEY", "your-secure-api-key")
# PORT = int(os.getenv("PORT", "8000"))
# SEED_DEMO = os.getenv("SEED_DEMO", "false").lower() == "true"
# SESSION_TTL_MIN = int(os.getenv("SESSION_TTL_MIN", "4320"))  # 3 days

# if not OPENAI_API_KEY:
#     raise RuntimeError("OPENAI_API_KEY missing")
# if not MONGODB_URI:
#     raise RuntimeError("MONGODB_URI missing")

# # ---------------------
# # Globals (thread-safe)
# # ---------------------
# client: Optional[AsyncIOMotorClient] = None
# openai_client: Optional[AsyncOpenAI] = None
# faiss_store: Optional[FAISS] = None

# db = None
# users_col = None
# conversations_col = None
# journals_col = None
# embeddings_col = None
# personalities_col = None
# errors_col = None
# saved_greetings_col = None
# greetings_cache_col = None
# relationships_col = None
# sessions_col = None

# mongo_lock = threading.Lock()
# openai_lock = threading.Lock()
# faiss_lock = threading.Lock()

# embedding_cache = TTLCache(maxsize=1000, ttl=3600)
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
# FAISS_DIR = "faiss_store_v1"
# watcher_task: Optional[asyncio.Task] = None

# def as_utc_aware(dt: Optional[datetime]) -> Optional[datetime]:
#     if dt is None:
#         return None
#     return dt if dt.tzinfo is not None else dt.replace(tzinfo=pytz.UTC)

# # ---------------------
# # Lazy init functions
# # ---------------------
# async def get_mongo_client() -> AsyncIOMotorClient:
#     global client, db, users_col, conversations_col, journals_col, embeddings_col, personalities_col
#     global errors_col, saved_greetings_col, greetings_cache_col, relationships_col, sessions_col
#     with mongo_lock:
#         if client is None:
#             client = AsyncIOMotorClient(
#                 MONGODB_URI,
#                 tls=True,
#                 tlsAllowInvalidCertificates=True,
#                 maxPoolSize=50,
#                 minPoolSize=5,
#                 maxIdleTimeMS=30000,
#                 tz_aware=True
#             )
#             db = client["LF"]
#             users_col = db["users"]
#             conversations_col = db["conversations"]
#             journals_col = db["journal_entries"]
#             embeddings_col = db["embeddings"]
#             personalities_col = db["personalities"]
#             errors_col = db["errors"]
#             saved_greetings_col = db["saved_greetings"]
#             greetings_cache_col = db["greetings"]
#             relationships_col = db["relationships"]
#             sessions_col = db["sessions"]

#     # indexes
#     await conversations_col.create_index([("user_id", 1), ("timestamp", -1)])
#     await conversations_col.create_index([("speaker_id", 1), ("target_id", 1), ("timestamp", -1)])
#     await conversations_col.create_index([("content", "text")])
#     await journals_col.create_index([("user_id", 1), ("timestamp", -1)])
#     await journals_col.create_index([("content", "text")])
#     await embeddings_col.create_index([("item_id", 1), ("item_type", 1)])
#     await personalities_col.create_index([("user_id", 1)])
#     await errors_col.create_index([("timestamp", -1)])
#     await saved_greetings_col.create_index([("target_id", 1), ("bot_role", 1), ("timestamp", -1)])
#     await greetings_cache_col.create_index([("key", 1), ("timestamp", -1)])
#     await relationships_col.create_index([("user_id", 1), ("other_user_id", 1)], unique=True)
#     await sessions_col.create_index("expires_at", expireAfterSeconds=0)

#     return client

# async def get_openai_client() -> AsyncOpenAI:
#     global openai_client
#     with openai_lock:
#         if openai_client is None:
#             openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
#     return openai_client

# async def ensure_faiss_store():
#     global faiss_store
#     with faiss_lock:
#         if faiss_store is None:
#             if os.path.isdir(FAISS_DIR):
#                 try:
#                     faiss_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
#                     return
#                 except Exception as e:
#                     logger.warning(f"FAISS load failed: {e}. Rebuilding...")
#     await initialize_faiss_store()

# async def initialize_faiss_store():
#     global faiss_store
#     await get_mongo_client()
#     with faiss_lock:
#         if os.path.isdir(FAISS_DIR):
#             try:
#                 faiss_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
#                 return
#             except Exception:
#                 pass

#     emb_data = await embeddings_col.find().to_list(length=None)
#     docs: List[Document] = []
#     for emb in emb_data:
#         try:
#             item_id = emb.get("item_id")
#             item_type = emb.get("item_type")
#             if not item_id or not item_type:
#                 await embeddings_col.delete_one({"_id": emb["_id"]})
#                 continue
#             col = conversations_col if item_type == "conversation" else journals_col
#             id_field = "conversation_id" if item_type == "conversation" else "entry_id"
#             base = await col.find_one({id_field: item_id})
#             if not base:
#                 await embeddings_col.delete_one({"item_id": item_id, "item_type": item_type})
#                 continue

#             content = emb.get("content", base.get("content", ""))
#             if not content:
#                 await embeddings_col.delete_one({"item_id": item_id, "item_type": item_type})
#                 continue

#             metadata = {
#                 "item_id": item_id,
#                 "item_type": item_type,
#                 "user_id": emb.get("user_id", []),
#                 "speaker_id": emb.get("speaker_id"),
#                 "target_id": emb.get("target_id"),
#                 "speaker_name": emb.get("speaker_name"),
#                 "target_name": emb.get("target_name"),
#                 "timestamp": as_utc_aware(emb.get("timestamp"))
#             }
#             docs.append(Document(page_content=content, metadata=metadata))
#         except Exception:
#             await embeddings_col.delete_one({"_id": emb["_id"]})

#     with faiss_lock:
#         if docs:
#             faiss_store = FAISS.from_documents(docs, embeddings)
#         else:
#             faiss_store = FAISS.from_texts(["empty"], embeddings)
#         faiss_store.save_local(FAISS_DIR)

# # ---------------------
# # Security: session tokens
# # ---------------------
# async def create_session(user_id: str) -> str:
#     await get_mongo_client()
#     token = str(uuid.uuid4())
#     now = datetime.now(pytz.UTC)
#     await sessions_col.insert_one({
#         "token": token,
#         "user_id": user_id,
#         "created_at": now,
#         "expires_at": now + timedelta(minutes=SESSION_TTL_MIN)
#     })
#     return token

# async def require_session(x_session_token: str = Header(...)) -> Dict[str, Any]:
#     await get_mongo_client()
#     sess = await sessions_col.find_one({"token": x_session_token})
#     if not sess:
#         raise HTTPException(status_code=401, detail="Invalid session")
#     if as_utc_aware(sess["expires_at"]) < datetime.now(pytz.UTC):
#         raise HTTPException(status_code=401, detail="Session expired")
#     user = await users_col.find_one({"user_id": sess["user_id"]})
#     if not user:
#         raise HTTPException(status_code=401, detail="User not found")
#     return {"token": x_session_token, "user": user}

# # ---------------------
# # Password hashing
# # ---------------------
# def hash_password(password: str) -> Dict[str, str]:
#     salt = secrets.token_bytes(16)
#     dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
#     return {"salt": base64.b64encode(salt).decode(), "hash": base64.b64encode(dk).decode()}

# def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
#     salt = base64.b64decode(salt_b64.encode())
#     expected = base64.b64decode(hash_b64.encode())
#     dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
#     return secrets.compare_digest(dk, expected)

# # ---------------------
# # Connection manager (WebSockets)
# # ---------------------
# class ConnectionManager:
#     def __init__(self):
#         self.active: Dict[str, WebSocket] = {}
#         self.lock = asyncio.Lock()

#     async def connect(self, user_id: str, websocket: WebSocket):
#         await websocket.accept()
#         async with self.lock:
#             self.active[user_id] = websocket

#     async def disconnect(self, user_id: str):
#         async with self.lock:
#             self.active.pop(user_id, None)

#     async def send_to(self, user_id: str, data: dict):
#         async with self.lock:
#             ws = self.active.get(user_id)
#         if ws:
#             await ws.send_json(data)

#     async def broadcast_presence(self):
#         async with self.lock:
#             online = list(self.active.keys())
#             sockets = list(self.active.values())
#         payload = {"type": "presence", "online": online}
#         for ws in sockets:
#             try:
#                 await ws.send_json(payload)
#             except Exception:
#                 pass

# manager = ConnectionManager()

# # ---------------------
# # FastAPI app (+ healthcheck)
# # ---------------------
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global watcher_task
#     await initialize_db()
#     watcher_task = asyncio.create_task(watch_collections())
#     yield
#     if watcher_task:
#         watcher_task.cancel()
#         try:
#             await watcher_task
#         except asyncio.CancelledError:
#             pass
#     if client:
#         client.close()

# app = FastAPI(title="Chatbot AI Twin API", lifespan=lifespan)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/healthz")
# async def healthz():
#     # Keep this independent of DB/OpenAI so Render health checks succeed
#     return {"ok": True}

# # ---------------------
# # Pydantic models
# # ---------------------
# class MessageRequest(BaseModel):
#     speaker_id: str
#     target_id: str
#     bot_role: Optional[str] = None
#     user_input: str

# class MessageResponse(BaseModel):
#     response: str
#     error: Optional[str] = None

# class SignupRequest(BaseModel):
#     username: str
#     display_name: str
#     password: str

# class LoginRequest(BaseModel):
#     username: str
#     password: str

# class RelationshipSetRequest(BaseModel):
#     other_user_id: str
#     relation: str

# class JournalAddRequest(BaseModel):
#     content: str
#     consent: bool

# # ---------------------
# # HTML UI (same as your enhanced version, with tiny fix: do not force bot_role)
# # ---------------------
# @app.get("/", response_class=HTMLResponse)
# async def home():
#     html = r"""
# <!DOCTYPE html>
# <html>
# <head>
# <meta charset="utf-8"/>
# <title>AI Twin Chat</title>
# <style>
# * { box-sizing: border-box }
# :root { --bg:#0f172a; --fg:#fff; --muted:#64748b; --border:#e2e8f0; --card:#f8fafc; }
# body { margin:0; font-family: Inter, system-ui, Arial }
# header { padding: 10px 16px; background:var(--bg); color:var(--fg); display:flex; justify-content:space-between; align-items:center }
# main { display:flex; height: calc(100vh - 56px) }
# #sidebar { width:360px; border-right:1px solid var(--border); padding:12px; overflow:auto }
# #content { flex:1; display:flex; flex-direction:column }
# section { margin-bottom:16px }
# h3 { margin:8px 0 }
# input, select, button, textarea { padding:10px; margin:6px 0; width:100%; font-size:14px }
# textarea { resize:none; min-height:48px; max-height:160px; line-height:20px; }
# button { cursor:pointer }
# .user-item { padding:8px; border:1px solid var(--border); border-radius:8px; margin:6px 0; display:flex; gap:8px; align-items:center; justify-content:space-between; background:#fff }
# .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#e2e8f0 }
# .badge.online { background:#bbf7d0 }
# .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#fde68a; color:#92400e; margin-left:6px }
# .chatgrid { flex:1; display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:12px; padding:12px; overflow:auto }
# .chatbox { border:1px solid var(--border); border-radius:10px; display:flex; flex-direction:column; min-height:360px; background:#fff }
# .chatbox header { background:var(--card); color:#0f172a; font-weight:600; padding:8px 10px; display:flex; align-items:center; justify-content:space-between }
# .header-right { display:flex; align-items:center; gap:6px }
# .messages { flex:1; padding:10px; overflow:auto; font-size:14px; background:#fafafa }
# .msg { margin:8px 0; padding:8px 10px; border-radius:8px; max-width:92%; box-shadow:0 1px 0 rgba(0,0,0,.04) }
# .msg.you { background:#dbeafe; align-self:flex-end }
# .msg.them { background:#dcfce7 }
# .msg.ai { background:#fef3c7 }
# .meta { font-size:11px; color:var(--muted); margin-top:4px }
# .typing { font-size:12px; color:var(--muted); margin:6px 0 0 2px }
# .actions { padding:8px; display:flex; gap:8px; border-top:1px solid var(--border); background:#fff }
# .actions textarea { flex:1; border:1px solid var(--border); border-radius:8px; padding:10px 12px; }
# .actions button { width:auto; padding:10px 14px; border-radius:8px; background:#0ea5e9; color:#fff; border:none }
# .actions button:disabled { opacity:.6; cursor:not-allowed }
# .small { font-size:12px; color:var(--muted) }
# .form-card { border:1px solid var(--border); border-radius:10px; padding:12px; background:#fff }
# .row { display:flex; gap:8px }
# .row > * { flex:1 }
# hr { border:0; border-top:1px solid var(--border); margin:12px 0 }
# .list { border:1px solid var(--border); border-radius:10px; padding:8px; background:#fff; max-height:220px; overflow:auto }
# .item { padding:6px 6px; border-bottom:1px dashed #e5e7eb }
# .item:last-child { border-bottom:none }
# .warn { color:#92400e; background:#fef3c7; padding:6px 8px; border-radius:8px; font-size:12px; }
# </style>
# </head>
# <body>
# <header>
#   <div>AI Twin Chat</div>
#   <div id="whoami" class="small"></div>
# </header>
# <main>
#   <div id="sidebar">
#     <section id="auth">
#       <div class="form-card">
#         <h3>Login</h3>
#         <input id="loginUsername" placeholder="username"/>
#         <input id="loginPassword" placeholder="password" type="password"/>
#         <button id="loginBtn">Login</button>
#       </div>
#       <hr/>
#       <div class="form-card">
#         <h3>Sign up</h3>
#         <input id="signupUsername" placeholder="username"/>
#         <input id="signupDisplayName" placeholder="display name"/>
#         <input id="signupPassword" placeholder="password" type="password"/>
#         <button id="signupBtn">Create account</button>
#       </div>
#       <div style="margin-top:10px">
#         <div class="small">Server key for UI (x-api-key)</div>
#         <input id="gatewayKey" placeholder="x-api-key (server key)" />
#       </div>
#     </section>

#     <section id="me" style="display:none">
#       <div class="form-card">
#         <h3>Me</h3>
#         <div id="meInfo"></div>
#         <div class="row">
#           <label class="small" style="display:flex; align-items:center; gap:8px">
#             <input type="checkbox" id="aiToggle"/>
#             AI respond for me
#           </label>
#           <button id="logoutBtn" style="background:#ef4444">Logout</button>
#         </div>
#       </div>
#     </section>

#     <section id="users" style="display:none">
#       <h3>Users</h3>
#       <div id="usersList"></div>
#     </section>

#     <section id="rel" style="display:none">
#       <h3>Set Relationship</h3>
#       <div class="row">
#         <select id="relOther"></select>
#         <select id="relKind">
#           <option>daughter</option><option>son</option><option>mother</option><option>father</option>
#           <option>sister</option><option>brother</option><option>wife</option><option>husband</option>
#           <option>friend</option>
#         </select>
#       </div>
#       <button id="relSave">Save</button>
#       <div id="relStatus" class="small"></div>
#     </section>

#     <!-- JOURNAL SECTION -->
#     <section id="journal" style="display:none">
#       <h3>Journal</h3>
#       <div class="warn">Notes here become your private memory and may be used to personalize replies.</div>
#       <textarea id="journalText" placeholder="Write a private note you'll want your AI Twin to remember…"></textarea>
#       <label class="small" style="display:flex; gap:8px; align-items:center; margin-top:6px">
#         <input type="checkbox" id="journalConsent"/> I understand and consent to this note being used to train my AI Twin.
#       </label>
#       <div class="row">
#         <button id="saveJournalBtn">Save Journal</button>
#         <button id="refreshJournalBtn" style="background:#64748b">Refresh</button>
#       </div>
#       <div id="journalStatus" class="small"></div>
#       <div style="margin-top:8px" class="small">Recent entries</div>
#       <div id="journalList" class="list"></div>
#     </section>
#   </div>

#   <div id="content">
#     <div class="chatgrid" id="chatGrid"></div>
#   </div>
# </main>

# <script>
# let API = location.origin;
# let API_KEY = "";
# let SESSION = "";
# let ME = null;
# let WS = null;
# const chatBoxes = new Map(); // other_user_id -> {box, area, btn, typingEl, aiExpected}

# function el(id){ return document.getElementById(id) }

# function setAuthVisible(loggedIn){
#   el('auth').style.display = loggedIn ? 'none' : 'block';
#   el('me').style.display = loggedIn ? 'block' : 'none';
#   el('users').style.display = loggedIn ? 'block' : 'none';
#   el('rel').style.display = loggedIn ? 'block' : 'none';
#   el('journal').style.display = loggedIn ? 'block' : 'none';
# }

# function autoresizeTA(ta){
#   ta.style.height = 'auto';
#   ta.style.height = Math.min(160, Math.max(48, ta.scrollHeight)) + 'px';
# }

# async function req(path, method='GET', body=null){
#   const headers = {'Content-Type':'application/json'};
#   if(API_KEY && API_KEY.toLowerCase()!=='disabled') headers['x-api-key']=API_KEY;
#   if(SESSION) headers['x-session-token']=SESSION;
#   const res = await fetch(API+path, {method, headers, body: body?JSON.stringify(body):undefined});
#   if(!res.ok){ throw new Error(await res.text()) }
#   return res.json();
# }

# function renderMe(){
#   el('meInfo').innerHTML = `
#     <div><b>${ME.display_name}</b> <span class="small">(@${ME.username})</span></div>
#     <div class="small">user_id: ${ME.user_id}</div>
#     <div class="small">AI: ${ME.ai_enabled ? 'ON' : 'OFF'}</div>
#   `;
#   el('whoami').innerText = `${ME.display_name} (@${ME.username})`;
#   el('aiToggle').checked = !!ME.ai_enabled;
# }

# async function refreshUsers(){
#   const data = await req('/users/list');
#   const container = el('usersList');
#   const relSel = el('relOther');
#   container.innerHTML = '';
#   relSel.innerHTML = '';
#   (data.users || []).filter(u=>u.user_id!==ME.user_id).forEach(u=>{
#      const div = document.createElement('div');
#      div.className='user-item';
#      const badge = `<span class="badge ${u.online?'online':''}">${u.online?'online':'offline'}</span>`;
#      const ai = u.ai_enabled ? `<span class="pill">AI replies</span>` : '';
#      div.innerHTML = `
#        <div>
#          <div><b>${u.display_name}</b> <span class="small">(@${u.username})</span> ${ai}</div>
#          <div class="small">rel: ${u.relation || '-'}</div>
#        </div>
#        <div>
#          ${badge}
#          <button data-id="${u.user_id}">Chat</button>
#        </div>`;
#      div.querySelector('button').onclick=()=>openChat(u);
#      container.appendChild(div);

#      const opt = document.createElement('option');
#      opt.value = u.user_id; opt.textContent = `${u.display_name} (@${u.username})`;
#      relSel.appendChild(opt);
#   });
# }

# function ensureBox(u){
#   if(chatBoxes.has(u.user_id)) return chatBoxes.get(u.user_id);
#   const div = document.createElement('div');
#   div.className='chatbox';
#   div.innerHTML = `
#     <header>
#       <div>${u.display_name} <span class="small">(@${u.username})</span></div>
#       <div class="header-right">
#         <span class="badge ${u.online?'online':''}" id="on_${u.user_id}">${u.online?'online':'offline'}</span>
#         ${u.ai_enabled?'<span class="pill">AI replies</span>':''}
#       </div>
#     </header>
#     <div class="messages" id="msg_${u.user_id}"></div>
#     <div class="typing" id="typing_${u.user_id}" style="display:none">AI is typing…</div>
#     <div class="actions">
#       <textarea placeholder="Write a message…" id="inp_${u.user_id}"></textarea>
#       <button id="send_${u.user_id}">Send</button>
#     </div>
#   `;
#   el('chatGrid').appendChild(div);
#   const area = el('inp_'+u.user_id);
#   const btn = el('send_'+u.user_id);
#   const typingEl = el('typing_'+u.user_id);
#   area.addEventListener('input', ()=>autoresizeTA(area));
#   area.addEventListener('keypress', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendTo(u.user_id); }});
#   btn.onclick=()=>sendTo(u.user_id);
#   chatBoxes.set(u.user_id, {box:div, area, btn, typingEl, aiExpected: !!u.ai_enabled});
#   return chatBoxes.get(u.user_id);
# }

# async function openChat(u){
#   const {area} = ensureBox(u);
#   // load last 30 messages
#   const res = await req(`/conversations/with/${u.user_id}?limit=30`);
#   const msgs = res.messages || [];
#   const pane = el('msg_'+u.user_id);
#   pane.innerHTML='';
#   msgs.forEach(m=>appendMsg(u.user_id, m));
#   area.focus();
# }

# function appendMsg(other_id, m, localEcho=false){
#   const pane = el('msg_'+other_id);
#   if(!pane) return;
#   const wrapper = document.createElement('div');
#   const who = localEcho ? 'you' : (m.speaker_id===ME.user_id ? 'you' : (m.source==='ai_twin' ? 'ai' : 'them'));
#   wrapper.className = `msg ${who}`;
#   const when = new Date(m.timestamp).toLocaleString();
#   wrapper.innerHTML = `${m.content}<div class="meta">${when}${localEcho?' • ✓ Sent':''}</div>`;
#   pane.appendChild(wrapper);
#   pane.scrollTop = pane.scrollHeight;
# }

# function showTyping(other_id, on){
#   const elT = el('typing_'+other_id);
#   if(!elT) return;
#   elT.style.display = on ? 'block' : 'none';
# }

# async function sendTo(other_id){
#   const ref = chatBoxes.get(other_id);
#   if(!ref) return;
#   const {area, btn, aiExpected} = ref;
#   const text = (area.value || '').trim();
#   if(!text) return;
#   area.value=''; autoresizeTA(area);
#   btn.disabled = true;

#   // Local echo immediately
#   appendMsg(other_id, {content:text, timestamp: new Date().toISOString(), speaker_id: ME.user_id, source:'human'}, true);

#   // If we expect AI, show typing until reply arrives
#   if(aiExpected) showTyping(other_id, true);

#   // Prefer WS if connected, else HTTP
#   if(WS && WS.readyState===1){
#     WS.send(JSON.stringify({type:'chat', to: other_id, text}));
#     // AI reply will arrive as WS "ai" event if enabled
#     btn.disabled = false;
#   }else{
#     try{
#       const res = await req('/send_message','POST', {
#         speaker_id: ME.user_id,
#         target_id: other_id,
#         user_input: text
#       });
#       // If server returned an AI reply (HTTP path), append it
#       if(res && res.response && res.response !== 'Sent.'){
#         appendMsg(other_id, {content: res.response, timestamp: new Date().toISOString(), speaker_id: other_id, source:'ai_twin'});
#       }
#     }catch(e){
#       appendMsg(other_id, {content: '⚠️ Failed to send: '+(e.message||e), timestamp: new Date().toISOString(), speaker_id: other_id, source:'system'});
#     }finally{
#       showTyping(other_id, false);
#       btn.disabled = false;
#     }
#   }
# }

# function connectWS(){
#   if(WS) try{ WS.close() }catch(e){}
#   const qp = new URLSearchParams({ token: SESSION, user_id: ME.user_id });
#   WS = new WebSocket(API.replace(/^http/,'ws')+'/ws?'+qp.toString());
#   WS.onmessage = (ev)=>{
#     try{
#       const msg = JSON.parse(ev.data);
#       if(msg.type==='presence'){
#         // Update online badges quickly
#         (msg.online||[]).forEach(uid=>{
#           const b = document.getElementById('on_'+uid);
#           if(b){ b.classList.add('online'); b.textContent='online'; }
#         });
#       }else if(msg.type==='chat'){
#         appendMsg(msg.from, msg.payload);
#       }else if(msg.type==='ai'){
#         appendMsg(msg.from, msg.payload);
#         showTyping(msg.from, false);
#       }
#     }catch(e){}
#   };
# }

# /* ---- JOURNAL UI wiring ---- */
# async function refreshJournal(){
#   try{
#     const res = await req('/journals/list','GET');
#     const host = el('journalList');
#     host.innerHTML = (res.entries||[]).map(e=>{
#       const when = new Date(e.timestamp).toLocaleString();
#       const safe = (e.content||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');
#       return `<div class="item"><div>${safe}</div><div class="small">${when}</div></div>`;
#     }).join('') || '<div class="small">No entries yet.</div>';
#   }catch(e){
#     el('journalStatus').innerText = 'Failed to load: '+(e.message||e);
#   }
# }

# async function saveJournal(){
#   const txt = (el('journalText').value||'').trim();
#   const consent = el('journalConsent').checked;
#   if(!txt){ el('journalStatus').innerText='Write something first.'; return; }
#   if(!consent){ el('journalStatus').innerText='Please check the consent box.'; return; }
#   el('journalStatus').innerText='Saving...';
#   try{
#     await req('/journals/add','POST',{content: txt, consent});
#     el('journalText').value=''; el('journalConsent').checked=false;
#     el('journalStatus').innerText='Saved!';
#     refreshJournal();
#   }catch(e){
#     el('journalStatus').innerText='Failed: '+(e.message||e);
#   }
# }

# document.addEventListener('DOMContentLoaded', ()=>{
#   // Server key persistence
#   el('gatewayKey').value = localStorage.getItem('gw') || '';
#   API_KEY = el('gatewayKey').value;
#   el('gatewayKey').addEventListener('change', ()=>{ API_KEY = el('gatewayKey').value; localStorage.setItem('gw', API_KEY) });

#   // Login
#   el('loginBtn').onclick = async ()=>{
#     API_KEY = el('gatewayKey').value;
#     const username = el('loginUsername').value.trim();
#     const password = el('loginPassword').value.trim();
#     const res = await req('/auth/login','POST',{username,password});
#     SESSION = res.token; ME = res.user;
#     setAuthVisible(true); renderMe(); connectWS(); await refreshUsers(); await refreshJournal();
#   };

#   // Signup
#   el('signupBtn').onclick = async ()=>{
#     API_KEY = el('gatewayKey').value;
#     const username = el('signupUsername').value.trim();
#     const display_name = el('signupDisplayName').value.trim() || username;
#     const password = el('signupPassword').value.trim();
#     if(!username || !password){ alert('Username and password required'); return; }
#     await req('/auth/signup','POST',{username,display_name,password});
#     alert('Signed up! Now login using the Login form above.');
#   };

#   // Logout
#   el('logoutBtn').onclick = async ()=>{
#     await req('/auth/logout','POST',{}); SESSION=''; ME=null; setAuthVisible(false); location.reload();
#   };

#   // AI toggle
#   el('aiToggle').onchange = async (e)=>{
#     await req(`/users/me/ai-toggle?enabled=${e.target.checked}`,'PATCH');
#     ME.ai_enabled = e.target.checked;
#   };

#   // Relationship save
#   el('relSave').onclick = async ()=>{
#     const other = el('relOther').value;
#     const relation = el('relKind').value;
#     if(!other) return;
#     await req('/relationships/set','POST', {other_user_id: other, relation});
#     el('relStatus').innerText = 'Saved!';
#     setTimeout(()=>{el('relStatus').innerText=''},1500);
#     await refreshUsers();
#   };

#   // Journal buttons
#   el('saveJournalBtn').onclick = saveJournal;
#   el('refreshJournalBtn').onclick = refreshJournal;

#   setAuthVisible(false);
# });
# </script>
# </body>
# </html>
#     """
#     return HTMLResponse(html)

# # ---------------------
# # Auth routes
# # ---------------------
# from typing import Optional as _Optional

# def require_api_key(x_api_key: _Optional[str] = Header(None)):
#     expected = (PUBLIC_UI_API_KEY or "").strip()
#     # If PUBLIC_UI_API_KEY empty or "disabled", skip the check
#     if expected and expected.lower() != "disabled":
#         if x_api_key != expected:
#             raise HTTPException(status_code=401, detail="Invalid API key")

# @app.post("/auth/signup")
# async def signup(req: SignupRequest, _: None = Depends(require_api_key)):
#     await get_mongo_client()
#     existing = await users_col.find_one({"username": req.username})
#     if existing:
#         raise HTTPException(status_code=400, detail="Username taken")
#     user_id = f"user_{uuid.uuid4().hex[:8]}"
#     h = hash_password(req.password)
#     now = datetime.now(pytz.UTC)
#     doc = {
#         "user_id": user_id,
#         "username": req.username,
#         "display_name": req.display_name,
#         "password_salt": h["salt"],
#         "password_hash": h["hash"],
#         "ai_enabled": False,
#         "created_at": now,
#         "last_seen": now
#     }
#     await users_col.insert_one(doc)
#     return {"ok": True, "user_id": user_id}

# @app.post("/auth/login")
# async def login(req: LoginRequest, _: None = Depends(require_api_key)):
#     await get_mongo_client()
#     user = await users_col.find_one({"username": req.username})
#     if not user:
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     if not verify_password(req.password, user.get("password_salt",""), user.get("password_hash","")):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     token = await create_session(user["user_id"])
#     await users_col.update_one({"user_id": user["user_id"]}, {"$set": {"last_seen": datetime.now(pytz.UTC)}})
#     return {"token": token, "user": {"user_id": user["user_id"], "username": user["username"], "display_name": user["display_name"], "ai_enabled": user.get("ai_enabled", False)}}

# @app.post("/auth/logout")
# async def logout(sess=Depends(require_session), _: None = Depends(require_api_key)):
#     await sessions_col.delete_one({"token": sess["token"]})
#     return {"ok": True}

# # ---------------------
# # Users / Relationships / Presence
# # ---------------------
# @app.get("/users/me")
# async def users_me(sess=Depends(require_session), _: None = Depends(require_api_key)):
#     u = sess["user"]
#     return {"user": {"user_id": u["user_id"], "username": u["username"], "display_name": u["display_name"], "ai_enabled": u.get("ai_enabled", False)}}

# @app.patch("/users/me/ai-toggle")
# async def toggle_ai(enabled: bool = Query(...), sess=Depends(require_session), _: None = Depends(require_api_key)):
#     await users_col.update_one({"user_id": sess["user"]["user_id"]}, {"$set": {"ai_enabled": bool(enabled)}})
#     return {"ok": True, "ai_enabled": bool(enabled)}

# @app.get("/users/list")
# async def users_list(sess=Depends(require_session), _: None = Depends(require_api_key)):
#     me_id = sess["user"]["user_id"]
#     all_users = await users_col.find({}, {"password_hash": 0, "password_salt": 0}).to_list(length=None)
#     rels = { r["other_user_id"]: r["relation"] async for r in relationships_col.find({"user_id": me_id}) }
#     async with manager.lock:
#         online = set(manager.active.keys())
#     users = []
#     for u in all_users:
#         users.append({
#             "user_id": u["user_id"],
#             "username": u["username"],
#             "display_name": u["display_name"],
#             "online": u["user_id"] in online,
#             "relation": rels.get(u["user_id"]),
#             "ai_enabled": bool(u.get("ai_enabled", False))
#         })
#     return {"users": users}

# # --- Relationship utils (NEW) ---
# INVERSE_REL = {
#     "mother": "son",
#     "father": "son",
#     "son": "father",
#     "daughter": "mother",
#     "sister": "brother",
#     "brother": "sister",
#     "wife": "husband",
#     "husband": "wife",
#     "friend": "friend"
# }

# async def resolve_target_role_for_reply(speaker_id: str, target_id: str) -> str:
#     await get_mongo_client()
#     rel = await relationships_col.find_one({"user_id": speaker_id, "other_user_id": target_id})
#     if rel and "relation" in rel:
#         return rel["relation"].lower()
#     return "friend"  

# @app.post("/relationships/set")
# async def rel_set(req: RelationshipSetRequest, sess=Depends(require_session), _: None = Depends(require_api_key)):
#     me_id = sess["user"]["user_id"]
#     now = datetime.now(pytz.UTC)
#     rel = req.relation.strip().lower()

#     # forward: me -> other
#     await relationships_col.update_one(
#         {"user_id": me_id, "other_user_id": req.other_user_id},
#         {"$set": {"relation": rel, "updated_at": now}},
#         upsert=True
#     )
#     # inverse: other -> me
#     inv = INVERSE_REL.get(rel, "friend")
#     await relationships_col.update_one(
#         {"user_id": req.other_user_id, "other_user_id": me_id},
#         {"$set": {"relation": inv, "updated_at": now}},
#         upsert=True
#     )
#     return {"ok": True}

# @app.get("/relationships/with/{other_id}")
# async def rel_get(other_id: str, sess=Depends(require_session), _: None = Depends(require_api_key)):
#     me_id = sess["user"]["user_id"]
#     r = await relationships_col.find_one({"user_id": me_id, "other_user_id": other_id})
#     return {"relation": (r or {}).get("relation")}

# # ---------------------
# # Core AI pieces
# # ---------------------
# def preprocess_input(user_input: str) -> str:
#     try:
#         doc = nlp(user_input)
#         key_terms = []
#         for t in doc:
#             if hasattr(t, "pos_") and hasattr(t, "is_stop"):
#                 if t.pos_ in ["NOUN", "VERB"] and not t.is_stop:
#                     key_terms.append(t.text.lower())
#             else:
#                 key_terms.append(t.text.lower())
#         extra_terms = []
#         for term in key_terms:
#             try:
#                 syns = wordnet.synsets(term)
#             except Exception:
#                 syns = []
#             synonyms = set()
#             for syn in syns:
#                 for lemma in syn.lemmas():
#                     w = lemma.name().replace('_',' ')
#                     if w != term and len(w.split()) <= 2:
#                         synonyms.add(w)
#             extra_terms.extend(list(synonyms)[:3])
#         if extra_terms:
#             user_input += " " + " ".join(set(extra_terms[:10]))
#         return user_input
#     except Exception:
#         return user_input

# async def get_recent_conversation_history(speaker_id: str, target_id: str, limit: int = 6) -> List[dict]:
#     await get_mongo_client()
#     pipeline = [
#         {"$match": {
#             "user_id": {"$all": [speaker_id, target_id]},
#             "$or": [{"speaker_id": speaker_id, "target_id": target_id},
#                     {"speaker_id": target_id, "target_id": speaker_id}]
#         }},
#         {"$sort": {"timestamp": -1}},
#         {"$limit": limit},
#         {"$sort": {"timestamp": 1}}
#     ]
#     out = []
#     async for conv in conversations_col.aggregate(pipeline):
#         sp_name = conv.get("speaker_name")
#         if not sp_name:
#             u = await users_col.find_one({"user_id": conv["speaker_id"]})
#             sp_name = (u or {}).get("display_name") or (u or {}).get("username") or conv["speaker_id"]
#         raw_ts = as_utc_aware(conv["timestamp"])
#         out.append({
#             "speaker": sp_name,
#             "content": conv["content"],
#             "timestamp": raw_ts.strftime("%Y-%m-%d %H:%M:%S"),
#             "type": conv.get("type","user_input"),
#             "source": conv.get("source", "human"),
#             "raw_timestamp": raw_ts,
#             "conversation_id": conv["conversation_id"]
#         })
#     return out

# async def generate_personality_traits(user_id: str) -> dict:
#     await get_mongo_client()
    
#     # Fetch conversations and journals
#     convs = [doc async for doc in conversations_col.find({"user_id": user_id}).sort("timestamp", -1).limit(500)]
#     journals = [doc async for doc in journals_col.find({"user_id": user_id}).sort("timestamp", -1).limit(500)]
#     data_text = "\n".join(
#         [f"[{doc['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {doc['content']}" for doc in convs] +
#         [f"[{doc['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] (Journal) {doc['content']}" for doc in journals]
#     )[:1500]  # Increased limit for richer analysis
    
#     if not data_text:
#         return {"core_traits": {}, "sub_traits": []}
    
#     # Check for cached traits
#     cached = await personalities_col.find_one({"user_id": user_id})
#     if cached and "traits" in cached:
#         return cached["traits"]

#     u = await users_col.find_one({"user_id": user_id})
#     display_name = (u or {}).get("display_name", user_id)
    
#     # Enhanced prompt for nuanced personality analysis
#     big_five_prompt = f"""
#     You are an expert psychologist analyzing the personality of {display_name} based on their conversation and journal entries.
#     Text data (conversations and journals, with timestamps):
#     {data_text}
    
#     Return a JSON object with:
#     - "core_traits": 5 Big Five traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) with:
#       - "score" (0-100)
#       - "explanation" (one sentence describing evidence from the text)
#     - "sub_traits": 3-5 unique traits (e.g., humorous, empathetic, curious) with:
#       - "trait" (name of the trait)
#       - "description" (one sentence explaining how it manifests in the text)
#       - "tone" (e.g., playful, nurturing, serious, to guide conversational style)
    
#     Ensure the traits reflect the user's conversational style, emotional tone, and context (e.g., family role, relationships).
#     Keep the response concise, within 700 tokens, and focus on specific evidence from the text.
#     """
    
#     traits = None
#     for attempt in range(3):
#         try:
#             resp = await (await get_openai_client()).chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {"role": "system", "content": "You are an expert psychologist generating detailed personality traits based on text data."},
#                     {"role": "user", "content": big_five_prompt}
#                 ],
#                 max_tokens=700,
#                 temperature=0.7
#             )
#             txt = resp.choices[0].message.content.strip()
#             txt = re.sub(r'^```json\s*|\s*```$', '', txt, flags=re.MULTILINE).strip()
#             traits = json.loads(txt)
#             if "core_traits" in traits and "sub_traits" in traits:
#                 if isinstance(traits["core_traits"], list):
#                     traits["core_traits"] = {t["trait"]: {"score": t["score"], "explanation": t["explanation"]} for t in traits["core_traits"]}
#                 break
#         except Exception as e:
#             logger.warning(f"Trait generation attempt {attempt + 1} failed: {e}")
#             if attempt == 2:
#                 traits = {
#                     "core_traits": {
#                         "Openness": {"score": 50, "explanation": "Neutral openness due to limited data."},
#                         "Conscientiousness": {"score": 50, "explanation": "Neutral conscientiousness due to limited data."},
#                         "Extraversion": {"score": 50, "explanation": "Neutral extraversion due to limited data."},
#                         "Agreeableness": {"score": 50, "explanation": "Neutral agreeableness due to limited data."},
#                         "Neuroticism": {"score": 50, "explanation": "Neutral neuroticism due to limited data."}
#                     },
#                     "sub_traits": [
#                         {"trait": "neutral", "description": "Shows balanced behavior due to limited data.", "tone": "neutral"},
#                         {"trait": "adaptable", "description": "Adapts to context based on available data.", "tone": "flexible"},
#                         {"trait": "curious", "description": "Engages with available information.", "tone": "inquisitive"}
#                     ]
#                 }
    
#     # Store traits in database
#     await personalities_col.update_one(
#         {"user_id": user_id},
#         {"$set": {"traits": traits, "updated_at": datetime.now(pytz.UTC)}},
#         upsert=True
#     )
#     return traits

# async def get_greeting_and_tone(bot_role: str, target_id: str) -> Tuple[str,str]:
#     await get_mongo_client()
#     key = f"greeting_{target_id}_{bot_role}"
#     cached = await greetings_cache_col.find_one({"key": key, "timestamp": {"$gte": datetime.now(pytz.UTC)-timedelta(hours=1)}})
#     if cached:
#         return cached["greeting"], cached["tone"]

#     saved = await saved_greetings_col.find_one({"target_id": target_id, "bot_role": bot_role.lower()}, sort=[("timestamp",-1)])
#     if saved:
#         return saved["greeting"], "warm, youthful" if bot_role.lower() in ["daughter","son"] else "nurturing, caring"

#     defaults = {
#         "daughter": ("Hey, Mom", "warm, youthful"),
#         "son": ("Hey, Mom", "warm, youthful"),
#         "mother": ("Hi, sweetie", "nurturing, caring"),
#         "father": ("Hey, kid", "warm, supportive"),
#         "sister": ("Yo, sis", "playful, casual"),
#         "brother": ("Yo, bro", "playful, casual"),
#         "wife": ("Hey, hon", "affectionate, conversational"),
#         "husband": ("Hey, hon", "affectionate, conversational"),
#         "friend": ("Hey, what's good?", "casual, friendly")
#     }
#     greeting, tone = defaults.get(bot_role.lower(), ("Hey","casual, friendly"))

#     traits = await generate_personality_traits(target_id)
#     prompt = f"""
#     You are generating a greeting for a {bot_role} with traits: {', '.join(traits.get('core_traits', {}).keys())}.
#     Return a JSON object: {{"greeting":"short greeting","tone":"tone description"}}
#     """
#     for attempt in range(3):
#         try:
#             resp = await (await get_openai_client()).chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {"role":"system","content":"Return only valid JSON with 'greeting' and 'tone' keys."},
#                     {"role":"user","content":prompt}
#                 ], max_tokens=100, temperature=0.5
#             )
#             txt = resp.choices[0].message.content.strip()
#             txt = re.sub(r'^```json\s*|\s*```$','',txt, flags=re.MULTILINE).strip()
#             obj = json.loads(txt)
#             if "greeting" in obj and "tone" in obj:
#                 greeting, tone = obj["greeting"], obj["tone"]
#                 break
#         except Exception:
#             if attempt==2: break

#     await greetings_cache_col.update_one({"key":key},{"$set":{"greeting":greeting,"tone":tone,"timestamp":datetime.now(pytz.UTC)}}, upsert=True)
#     return greeting, tone

# # ---------------------
# # RAG: memories (convos + journals)
# # ---------------------
# async def find_relevant_memories(speaker_id: str, user_id: str, user_input: str, speaker_name: str, max_memories: int = 5) -> List[dict]:
#     global faiss_store
#     await ensure_faiss_store()
#     await get_mongo_client()
#     loop = asyncio.get_event_loop()
#     processed = await loop.run_in_executor(None, preprocess_input, user_input)
#     cache_key = f"input_{hash(processed)}"
#     if cache_key in embedding_cache:
#         _ = embedding_cache[cache_key]
#     else:
#         _ = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
#         embedding_cache[cache_key] = _

#     udoc = await users_col.find_one({"user_id": user_id})
#     target_name = (udoc or {}).get("display_name") or (udoc or {}).get("username") or user_id

#     results = await loop.run_in_executor(None, lambda: faiss_store.similarity_search_with_score(processed, k=max_memories*3))
#     mems = []
#     for doc, score in results:
#         md = doc.metadata
#         item_id = md.get("item_id"); item_type = md.get("item_type")
#         if not item_id or not item_type: continue
#         col = conversations_col if item_type=="conversation" else journals_col
#         id_field = "conversation_id" if item_type=="conversation" else "entry_id"
#         q = {id_field:item_id, "user_id": user_id}
#         base = await col.find_one(q)
#         if not base:
#             await embeddings_col.delete_one({"item_id": item_id, "item_type": item_type})
#             continue
#         if item_type=="journal":
#             base["speaker_name"] = target_name

#         adjusted = 1.0 - score
#         if item_type=="journal": adjusted += 0.9
#         elif md.get("speaker_id")==speaker_id or md.get("target_id")==user_id: adjusted += 0.7
#         if speaker_name.lower() in base.get("content","").lower() or target_name.lower() in base.get("content","").lower():
#             adjusted += 0.3
#         ts = as_utc_aware(md.get("timestamp")) or as_utc_aware(base.get("timestamp"))
#         days_old = (datetime.now(pytz.UTC) - ts).days if ts else 9999
#         temporal_weight = 1/(1 + np.log1p(max(days_old,1)/30))
#         adjusted *= temporal_weight
#         if adjusted < 0.3: continue
#         mems.append({
#             "type": item_type, "content": base["content"], "timestamp": as_utc_aware(base["timestamp"]),
#             "score": float(adjusted), "user_id": md.get("user_id", []),
#             "speaker_id": md.get("speaker_id"), "speaker_name": base.get("speaker_name", target_name),
#             "target_id": md.get("target_id"), "target_name": md.get("target_name")
#         })
#     mems.sort(key=lambda x: x["score"], reverse=True)
#     return mems[:max_memories]

# async def should_include_memories(user_input: str, speaker_id: str, user_id: str) -> Tuple[bool, List[dict]]:
#     sp = await users_col.find_one({"user_id": speaker_id})
#     speaker_name = (sp or {}).get("display_name") or (sp or {}).get("username") or speaker_id
#     mems = await find_relevant_memories(speaker_id, user_id, user_input, speaker_name, max_memories=10)
#     if not mems: return False, []
#     loop = asyncio.get_event_loop()
#     processed = await loop.run_in_executor(None, preprocess_input, user_input)
#     inp = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
#     rel = []
#     for m in mems:
#         emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(m["content"]))
#         sim = np.dot(inp, emb) / (np.linalg.norm(inp)*np.linalg.norm(emb))
#         if sim >= 0.5:
#             rel.append(m)
#     return (len(rel)>0), rel[:3]

# # ---------------------
# # initialize_bot (role auto-detect)
# # ---------------------
# async def initialize_bot(speaker_id: str, target_id: str, bot_role: Optional[str], user_input: str) -> Tuple[str, str, bool, dict]:
#     sp = await users_col.find_one({"user_id": speaker_id})
#     tg = await users_col.find_one({"user_id": target_id})
#     if not sp or not tg:
#         raise ValueError("Invalid IDs")

#     # Auto-resolve role if not provided or unhelpful
#     role_in = (bot_role or "").strip().lower()
#     if not role_in or role_in == "friend":
#         role_in = await resolve_target_role_for_reply(speaker_id, target_id)

#     traits = await generate_personality_traits(target_id)
#     recent = await get_recent_conversation_history(speaker_id, target_id)

#     history_for_prompt = recent[:]
#     if recent:
#         last = recent[-1]
#         if last.get("content", "").strip() == user_input.strip():
#             history_for_prompt = recent[:-1]

#     allow_repeat_ref = False
#     try:
#         loop = asyncio.get_event_loop()
#         q_emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(user_input))
#         for m in history_for_prompt[-10:]:
#             if not m.get("content"): continue
#             emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(m["content"]))
#             sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb)))
#             if sim >= 0.92:
#                 allow_repeat_ref = True
#                 break
#     except Exception:
#         allow_repeat_ref = False

#     if history_for_prompt:
#         hist_text = "\n".join([f"[{m['raw_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {m['content']}" for m in history_for_prompt])
#         last_ts = history_for_prompt[-1]["raw_timestamp"]
#     else:
#         hist_text = "No earlier messages."
#         last_ts = None

#     use_greeting = (not history_for_prompt) or (datetime.now(pytz.UTC) - as_utc_aware(last_ts)).total_seconds() / 60 > 30
#     greeting, tone = await get_greeting_and_tone(role_in, target_id)

#     include, mems = await should_include_memories(user_input, speaker_id, target_id)
#     mems_text = "No relevant memories."
#     if include and mems:
#         good = [m for m in mems if all(k in m for k in ["content", "type", "timestamp", "speaker_name"])]
#         if good:
#             mems_text = "\n".join([f"- [{m['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {m['content']} ({m['type']}, said by {m['speaker_name']})" for m in good])

#     rails = f"""
#     Grounding rules:
#     - You may reference dates/timestamps in the earlier conversation history.
#     - Do NOT refer to the current message as if it were a past event.
#     - Only say "you asked this before..." if there is a clearly earlier, highly similar message. Permission: {"ALLOWED" if allow_repeat_ref else "NOT ALLOWED"}.
#     - If NOT ALLOWED, avoid implying repetition; respond normally.
#     """

#     trait_str = ', '.join([f"{k} ({v['explanation']})" for k, v in list(traits.get('core_traits', {}).items())[:3]]) or "balanced"
#     sp_name = (sp or {}).get("display_name") or (sp or {}).get("username") or speaker_id
#     tg_name = (tg or {}).get("display_name") or (tg or {}).get("username") or target_id

#     base_prompt = f"""
#     You are {tg_name}, responding as an AI Twin to {sp_name}, their {role_in}.
#     Use a {tone} tone and reflect your personality: {trait_str}.

#     Earlier conversation (timestamps included, excludes the current message):
#     {hist_text}

#     {rails}

#     - {'Start with "' + greeting + '" if no earlier messages or time gap > 30 minutes.' if use_greeting else 'Do not start with a greeting.'}
#     - Keep it short (2–3 sentences), natural, and personalized.
#     Current user input: {user_input}

#     Respond directly to the Current user input above.
#     """

#     if include:
#         base_prompt = base_prompt.replace("{rails}\n\n", "{rails}\n\nPotentially relevant memories:\n" + mems_text + "\n\n")

#     return base_prompt, greeting, use_greeting, traits

# async def generate_response(prompt: str, user_input: str, greeting: str, use_greeting: bool, speaker_id: str, target_id: str, role_in: str, traits: dict) -> str:
#     try:
#         # Summarize memories for conciseness
#         include, mems = await should_include_memories(user_input, speaker_id, target_id)
#         mems_summary = "No relevant memories."
#         if include and mems:
#             good = [m for m in mems if all(k in m for k in ["content", "type", "timestamp", "speaker_name"])]
#             if good:
#                 mems_summary = "\n".join([f"- {m['speaker_name']} said: '{m['content'][:50]}...' ({m['type']}, {m['timestamp'].strftime('%Y-%m-%d')})" for m in good[:3]])

#         # Enhance prompt with traits and memories
#         sub_traits = ", ".join([f"{t['trait']} ({t['tone']})" for t in traits.get('sub_traits', [])]) or "balanced"
#         enhanced_prompt = prompt.replace("{rails}\n\n", f"{rails}\n\nRelevant memories (summarized):\n{mems_summary}\n\n")
#         enhanced_prompt += f"""
#         - Match the response tone and style to {target_id}'s personality: {sub_traits}.
#         - Reflect their role as {role_in} and emotional tone from sub-traits.
#         - Respond in 2-3 sentences, staying natural and contextually relevant.
#         """

#         resp = await (await get_openai_client()).chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": f"You are an AI Twin mimicking {target_id}'s personality, tone, and style based on their learned traits and memories."},
#                 {"role": "user", "content": enhanced_prompt}
#             ],
#             max_tokens=200,
#             temperature=0.6
#         )
#         text = resp.choices[0].message.content.strip()
#         if len(text.split()) >= 4 and ((use_greeting and text.lower().startswith(greeting.lower())) or not use_greeting):
#             parts = text.split('. ')[:3]
#             text = '. '.join([p for p in parts if p]).strip()
#             if text and not text.endswith('.'): text += '.'
#             return text
#     except Exception as e:
#         await errors_col.insert_one({"error": str(e), "input": user_input, "timestamp": datetime.now(pytz.UTC)})
#     return f"{greeting}, sounds cool! What's up?" if use_greeting else "Sounds cool! What's up?"

# # ---------------------
# # Save message helper
# # ---------------------
# async def save_and_embed_message(speaker_id: str, target_id: str, text: str, source: str) -> dict:
#     await get_mongo_client()
#     await ensure_faiss_store()
#     sp = await users_col.find_one({"user_id": speaker_id})
#     tg = await users_col.find_one({"user_id": target_id})
#     sp_name = (sp or {}).get("display_name") or (sp or {}).get("username") or speaker_id
#     tg_name = (tg or {}).get("display_name") or (tg or {}).get("username") or target_id
#     now = datetime.now(pytz.UTC)
#     conv_id = str(uuid.uuid4())
#     doc = {
#         "conversation_id": conv_id,
#         "user_id": [speaker_id, target_id],
#         "speaker_id": speaker_id,
#         "speaker_name": sp_name,
#         "target_id": target_id,
#         "target_name": tg_name,
#         "content": text,
#         "type": "user_input" if source=="human" else "response",
#         "source": source,
#         "timestamp": now
#     }
#     await conversations_col.insert_one(doc)

#     processed = preprocess_input(text)
#     loop = asyncio.get_event_loop()
#     emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
#     await embeddings_col.insert_one({
#         "item_id": conv_id, "item_type":"conversation", "user_id":[speaker_id,target_id],
#         "speaker_id": speaker_id, "speaker_name": sp_name,
#         "target_id": target_id, "target_name": tg_name,
#         "embedding": emb, "timestamp": now, "content": text
#     })
#     try:
#         db_doc = Document(page_content=text, metadata={
#             "item_id": conv_id, "item_type":"conversation", "user_id":[speaker_id,target_id],
#             "speaker_id": speaker_id, "speaker_name": sp_name,
#             "target_id": target_id, "target_name": tg_name,
#             "timestamp": now
#         })
#         with faiss_lock:
#             faiss_store.add_documents([db_doc])
#             faiss_store.save_local(FAISS_DIR)
#     except Exception as e:
#         logger.warning(f"FAISS add fail: {e}")

#     return doc

# # ---------------------
# # HTTP Chat
# # ---------------------
# def require_api_and_session(sess=Depends(require_session), _: None = Depends(require_api_key)):
#     return sess

# @app.post("/send_message", response_model=MessageResponse)
# async def send_message(req: MessageRequest, sess=Depends(require_api_and_session)):
#     if sess["user"]["user_id"] != req.speaker_id:
#         raise HTTPException(status_code=403, detail="Sender mismatch")
#     await save_and_embed_message(req.speaker_id, req.target_id, req.user_input, source="human")
#     tg = await users_col.find_one({"user_id": req.target_id})
#     if tg and tg.get("ai_enabled", False):
#         prompt, greeting, use_greeting, traits = await initialize_bot(req.speaker_id, req.target_id, getattr(req, "bot_role", None), req.user_input)
#         ai_text = await generate_response(prompt, req.user_input, greeting, use_greeting, req.speaker_id, req.target_id, getattr(req, "bot_role", None), traits)
#         await save_and_embed_message(req.target_id, req.speaker_id, ai_text, source="ai_twin")
#         return MessageResponse(response=ai_text)
#     return MessageResponse(response="Sent.")

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     token = websocket.query_params.get("token")
#     user_id = websocket.query_params.get("user_id")
#     await get_mongo_client()
#     sess = await sessions_col.find_one({"token": token, "user_id": user_id})
#     if not sess:
#         await websocket.close(code=4401)
#         return

#     try:
#         await manager.connect(user_id, websocket)
#         await manager.broadcast_presence()
#         await users_col.update_one({"user_id": user_id}, {"$set": {"last_seen": datetime.now(pytz.UTC)}})
#         while True:
#             data = await websocket.receive_text()
#             try:
#                 msg = json.loads(data)
#             except Exception:
#                 continue
#             if msg.get("type") == "chat":
#                 to = msg["to"]
#                 text = msg["text"]
#                 saved = await save_and_embed_message(user_id, to, text, source="human")
#                 await manager.send_to(to, {"type": "chat", "from": user_id, "payload": {
#                     "speaker_id": saved["speaker_id"],
#                     "target_id": saved["target_id"],
#                     "content": saved["content"],
#                     "source": "human",
#                     "timestamp": saved["timestamp"].isoformat()
#                 }})
#                 tgt = await users_col.find_one({"user_id": to})
#                 if tgt and tgt.get("ai_enabled", False):
#                     prompt, greeting, use_greeting, traits = await initialize_bot(user_id, to, None, text)
#                     ai_text = await generate_response(prompt, text, greeting, use_greeting, user_id, to, None, traits)
#                     ai_saved = await save_and_embed_message(to, user_id, ai_text, source="ai_twin")
#                     await manager.send_to(user_id, {"type": "ai", "from": to, "payload": {
#                         "speaker_id": ai_saved["speaker_id"],
#                         "target_id": ai_saved["target_id"],
#                         "content": ai_saved["content"],
#                         "source": "ai_twin",
#                         "timestamp": ai_saved["timestamp"].isoformat()
#                     }})
#     except WebSocketDisconnect:
#         pass
#     finally:
#         await manager.disconnect(user_id)
#         await manager.broadcast_presence()

# @app.get("/conversations/with/{other_id}")
# async def history_with(other_id: str, limit: int = 30, sess=Depends(require_api_and_session)):
#     me = sess["user"]["user_id"]
#     cur = conversations_col.find({"user_id": {"$all":[me, other_id]}}).sort("timestamp",-1).limit(limit)
#     out = []
#     async for c in cur:
#         out.append({
#             "conversation_id": c["conversation_id"],
#             "speaker_id": c["speaker_id"],
#             "target_id": c["target_id"],
#             "content": c["content"],
#             "source": c.get("source","human"),
#             "timestamp": as_utc_aware(c["timestamp"]).isoformat()
#         })
#     return {"messages": list(reversed(out))}

# # ---------------------
# # Journal endpoints
# # ---------------------
# @app.post("/journals/add")
# async def journals_add(req: JournalAddRequest, sess=Depends(require_api_and_session)):
#     if not req.consent:
#         raise HTTPException(status_code=400, detail="Consent required: please confirm the checkbox.")
#     await get_mongo_client()
#     now = datetime.now(pytz.UTC)
#     entry_id = str(uuid.uuid4())
#     doc = {
#         "entry_id": entry_id,
#         "user_id": [sess["user"]["user_id"]],
#         "content": (req.content or "").strip(),
#         "timestamp": now
#     }
#     await journals_col.insert_one(doc)
#     try:
#         await process_new_entry(item_id=entry_id, item_type="journal", content=doc["content"], user_id=doc["user_id"])
#     except Exception:
#         pass
#     return {"ok": True, "entry_id": entry_id, "timestamp": now.isoformat()}

# @app.get("/journals/list")
# async def journals_list(limit: int = 20, sess=Depends(require_api_and_session)):
#     me = sess["user"]["user_id"]
#     cur = journals_col.find({"user_id": me}).sort("timestamp",-1).limit(limit)
#     out = []
#     async for j in cur:
#         out.append({
#             "entry_id": j["entry_id"],
#             "content": j.get("content",""),
#             "timestamp": as_utc_aware(j.get("timestamp")).isoformat() if j.get("timestamp") else None
#         })
#     return {"entries": out}

# # ---------------------
# # WebSocket Chat
# # ---------------------


# # ---------------------
# # Change streams
# # ---------------------
# async def process_new_entry(item_id: str, item_type: str, content: str, user_id: list,
#                             speaker_id: Optional[str] = None, speaker_name: Optional[str] = None,
#                             target_id: Optional[str] = None, target_name: Optional[str] = None):
#     try:
#         await get_mongo_client()
#         await ensure_faiss_store()
#         processed = preprocess_input(content)
#         loop = asyncio.get_event_loop()
#         emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
#         now = datetime.now(pytz.UTC)
#         doc = {
#             "item_id": item_id, "item_type": item_type, "user_id": user_id,
#             "content": content, "embedding": emb, "timestamp": now
#         }
#         if item_type=="conversation":
#             doc.update({"speaker_id": speaker_id, "speaker_name": speaker_name, "target_id": target_id, "target_name": target_name})
#         await embeddings_col.insert_one(doc)

#         with faiss_lock:
#             if faiss_store is None:
#                 faiss_store = FAISS.from_texts(["empty"], embeddings)
#             meta = {"item_id": item_id, "item_type": item_type, "user_id": user_id, "timestamp": now}
#             if item_type=="conversation":
#                 meta.update({"speaker_id": speaker_id, "speaker_name": speaker_name, "target_id": target_id, "target_name": target_name})
#             faiss_store.add_documents([Document(page_content=content, metadata=meta)])
#             faiss_store.save_local(FAISS_DIR)
#     except Exception as e:
#         await errors_col.insert_one({"error": str(e), "item_id": item_id, "item_type": item_type, "timestamp": datetime.now(pytz.UTC)})

# async def watch_conversations():
#     while True:
#         try:
#             await get_mongo_client()
#             async with conversations_col.watch([{"$match":{"operationType":"insert"}}], full_document="updateLookup") as stream:
#                 async for change in stream:
#                     doc = change["fullDocument"]
#                     if doc.get("type") == "user_input" and doc.get("source") == "human":
#                         await process_new_entry(
#                             item_id=doc["conversation_id"], item_type="conversation",
#                             content=doc["content"], user_id=doc["user_id"],
#                             speaker_id=doc.get("speaker_id"), speaker_name=doc.get("speaker_name"),
#                             target_id=doc.get("target_id"), target_name=doc.get("target_name")
#                         )
#         except Exception:
#             await errors_col.insert_one({"error": "watch_conversations error", "timestamp": datetime.now(pytz.UTC)})
#             await asyncio.sleep(5)

# async def watch_journals():
#     while True:
#         try:
#             await get_mongo_client()
#             async with journals_col.watch([{"$match":{"operationType":"insert"}}], full_document="updateLookup") as stream:
#                 async for change in stream:
#                     doc = change["fullDocument"]
#                     await process_new_entry(item_id=doc["entry_id"], item_type="journal", content=doc["content"], user_id=doc["user_id"])
#         except Exception:
#             await errors_col.insert_one({"error": "watch_journals error", "timestamp": datetime.now(pytz.UTC)})
#             await asyncio.sleep(5)

# async def watch_collections():
#     await asyncio.gather(watch_conversations(), watch_journals())

# # ---------------------
# # Demo seed / initialization
# # ---------------------
# async def clear_database():
#     await get_mongo_client()
#     await users_col.delete_many({})
#     await conversations_col.delete_many({})
#     await journals_col.delete_many({})
#     await embeddings_col.delete_many({})
#     await relationships_col.delete_many({})
#     await sessions_col.delete_many({})

# async def populate_users():
#     now = datetime.now(pytz.UTC)
#     def mkuser(uid, uname, name):
#         h=hash_password("password")
#         return {"user_id": uid, "username": uname, "display_name": name, "password_salt":h["salt"], "password_hash":h["hash"], "ai_enabled": False, "created_at": now, "last_seen": now}
#     base = [
#         mkuser("user1","nipa","Nipa"),
#         mkuser("user2","nick","Nick"),
#         mkuser("user3","arif","Arif"),
#         mkuser("user4","diana","Diana")
#     ]
#     for u in base:
#         if not await users_col.find_one({"user_id": u["user_id"]}):
#             await users_col.insert_one(u)

# async def batch_embed_texts(texts: List[str]):
#     try:
#         loop = asyncio.get_event_loop()
#         return await loop.run_in_executor(None, lambda: embeddings.embed_documents(texts))
#     except Exception:
#         return [None]*len(texts)

# async def populate_conversations():
#     now = datetime.now(pytz.UTC)
#     convs = [
#         # Nipa (user1) to Nick (user2)
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user1", "user2"], "speaker_id": "user1", "speaker_name": "Nipa", "target_id": "user2", "target_name": "Nick", "content": "Nick, I found this amazing book about fantasy worlds—wanna read it with me?", "type": "user_input", "source": "human", "timestamp": now - timedelta(days=2)},
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user2", "user1"], "speaker_id": "user2", "speaker_name": "Nick", "target_id": "user1", "target_name": "Nipa", "content": "Yo, sis, only if it’s got epic battles! Got any new games to try?", "type": "user_input", "source": "human", "timestamp": now - timedelta(days=2, hours=1)},
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user1", "user2"], "speaker_id": "user1", "speaker_name": "Nipa", "target_id": "user2", "target_name": "Nick", "content": "I’m kinda nervous about my art presentation tomorrow—any tips?", "type": "user_input", "source": "human", "timestamp": now - timedelta(days=1)},
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user2", "user1"], "speaker_id": "user2", "speaker_name": "Nick", "target_id": "user1", "target_name": "Nipa", "content": "Chill, Nipa, just be yourself—you’ll crush it!", "type": "user_input", "source": "human", "timestamp": now - timedelta(days=1, hours=1)},
#         # Arif (user3) to Diana (user4)
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user3", "user4"], "speaker_id": "user3", "speaker_name": "Arif", "target_id": "user4", "target_name": "Diana", "content": "Hon, let’s plan a family picnic this weekend—maybe by the lake?", "type": "user_input", "source": "human", "timestamp": now - timedelta(days=3)},
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user4", "user3"], "speaker_id": "user4", "speaker_name": "Diana", "target_id": "user3", "target_name": "Arif", "content": "That sounds wonderful, Arif—let’s bring some homemade lemonade!", "type": "user_input", "source": "human", "timestamp": now - timedelta(days=3, hours=1)},
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user3", "user4"], "speaker_id": "user3", "speaker_name": "Arif", "target_id": "user4", "target_name": "Diana", "content": "Work’s been tough—any chance we can sneak away for a date night?", "type": "user_input", "source": "human", "timestamp": now - timedelta(days=2)},
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user4", "user3"], "speaker_id": "user4", "speaker_name": "Diana", "target_id": "user3", "target_name": "Arif", "content": "Oh, hon, let’s do it—maybe that new Italian place?", "type": "user_input", "source": "human", "timestamp": now - timedelta(days=2, hours=1)},
#         # Nipa (user1) to Arif (user3)
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user1", "user3"], "speaker_id": "user1", "speaker_name": "Nipa", "target_id": "user3", "target_name": "Arif", "content": "Dad, I’m stressed about my art project—can you help me brainstorm?", "type": "user_input", "source": "human", "timestamp": now - timedelta(hours=12)},
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user3", "user1"], "speaker_id": "user3", "speaker_name": "Arif", "target_id": "user1", "target_name": "Nipa", "content": "Nipa, let’s sketch out some ideas together—grab your pencils!", "type": "user_input", "source": "human", "timestamp": now - timedelta(hours=11)},
#         # Diana (user4) to Nick (user2)
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user4", "user2"], "speaker_id": "user4", "speaker_name": "Diana", "target_id": "user2", "target_name": "Nick", "content": "Nick, sweetie, have you finished your homework yet?", "type": "user_input", "source": "human", "timestamp": now - timedelta(hours=10)},
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user2", "user4"], "speaker_id": "user2", "speaker_name": "Nick", "target_id": "user4", "target_name": "Diana", "content": "Mom, almost done—just stuck on this math problem!", "type": "user_input", "source": "human", "timestamp": now - timedelta(hours=9)},
#         # Nick (user2) to Arif (user3)
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user2", "user3"], "speaker_id": "user2", "speaker_name": "Nick", "target_id": "user3", "target_name": "Arif", "content": "Dad, can we get a new gaming console? The old one’s lagging!", "type": "user_input", "source": "human", "timestamp": now - timedelta(hours=8)},
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user3", "user2"], "speaker_id": "user3", "speaker_name": "Arif", "target_id": "user2", "target_name": "Nick", "content": "Nick, let’s check the budget—maybe if you ace that test!", "type": "user_input", "source": "human", "timestamp": now - timedelta(hours=7)},
#         # Diana (user4) to Nipa (user1)
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user4", "user1"], "speaker_id": "user4", "speaker_name": "Diana", "target_id": "user1", "target_name": "Nipa", "content": "Nipa, want to bake some cookies tonight? It’ll be fun!", "type": "user_input", "source": "human", "timestamp": now - timedelta(hours=6)},
#         {"conversation_id": str(uuid.uuid4()), "user_id": ["user1", "user4"], "speaker_id": "user1", "speaker_name": "Nipa", "target_id": "user4", "target_name": "Diana", "content": "Yes, Mom, let’s make chocolate chip—my favorite!", "type": "user_input", "source": "human", "timestamp": now - timedelta(hours=5)}
#     ]
#     for c in convs:
#         if not await conversations_col.find_one({"conversation_id": c["conversation_id"]}):
#             await conversations_col.insert_one(c)
#     embeddings_result = await batch_embed_texts([c["content"] for c in convs])
#     docs = []
#     for c, e in zip(convs, embeddings_result):
#         if e is not None and not await embeddings_col.find_one({"item_id": c["conversation_id"], "item_type": "conversation"}):
#             docs.append({
#                 "item_id": c["conversation_id"],
#                 "item_type": "conversation",
#                 "user_id": c["user_id"],
#                 "content": c["content"],
#                 "embedding": e,
#                 "timestamp": c["timestamp"],
#                 "speaker_id": c["speaker_id"],
#                 "speaker_name": c["speaker_name"],
#                 "target_id": c["target_id"],
#                 "target_name": c["target_name"]
#             })
#     if docs:
#         await embeddings_col.insert_many(docs)

# async def populate_journals():
#     now = datetime.now(pytz.UTC)
#     journals = [
#         # Nipa (user1)
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user1"], "content": "I’m so inspired by this fantasy novel—it feels like I’m in another world!", "timestamp": now - timedelta(hours=8)},
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user1"], "content": "Feeling nervous about my art presentation, but I’m excited to share my work.", "timestamp": now - timedelta(hours=6)},
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user1"], "content": "Sometimes I worry about fitting in at school, but Mom’s support makes it easier.", "timestamp": now - timedelta(hours=4)},
#         # Nick (user2)
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user2"], "content": "Just beat my high score in this new game—feeling unstoppable!", "timestamp": now - timedelta(hours=7)},
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user2"], "content": "School was annoying today—too much homework, not enough gaming time.", "timestamp": now - timedelta(hours=5)},        
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user2"], "content": "Kinda stoked about the science fair—might build a cool robot!", "timestamp": now - timedelta(hours=3)},
#         # Arif (user3)
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user3"], "content": "Work was stressful, but coming home to Diana and the kids is my safe haven.", "timestamp": now - timedelta(hours=9)},
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user3"], "content": "Thinking about planning a family camping trip—need to make time for it.", "timestamp": now - timedelta(hours=2)},
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user3"], "content": "Proud of Nipa’s art passion—she reminds me to stay creative too.", "timestamp": now - timedelta(hours=1)},
#         # Diana (user4)
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user4"], "content": "Nipa’s growing up so fast—it’s bittersweet watching her chase her dreams.", "timestamp": now - timedelta(hours=10)},
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user4"], "content": "Tried a new painting technique today—felt so alive creating something new!", "timestamp": now - timedelta(hours=4)},
#         {"entry_id": str(uuid.uuid4()), "user_id": ["user4"], "content": "Worried about Nick’s grades, but I know he’ll pull through with some encouragement.", "timestamp": now - timedelta(hours=2)}
#     ]
#     for j in journals:
#         if not await journals_col.find_one({"entry_id": j["entry_id"]}):
#             await journals_col.insert_one(j)
#     embeddings_result = await batch_embed_texts([j["content"] for j in journals])
#     docs = []
#     for j, e in zip(journals, embeddings_result):
#         if e is not None and not await embeddings_col.find_one({"item_id": j["entry_id"], "item_type": "journal"}):
#             docs.append({
#                 "item_id": j["entry_id"],
#                 "item_type": "journal",
#                 "user_id": j["user_id"],
#                 "content": j["content"],
#                 "embedding": e,
#                 "timestamp": j["timestamp"]
#             })
#     if docs:
#         await embeddings_col.insert_many(docs)



# async def verify_data():
#     counts = {
#         "Users": await users_col.count_documents({}),
#         "Conversations": await conversations_col.count_documents({}),
#         "Journals": await journals_col.count_documents({}),
#         "Embeddings": await embeddings_col.count_documents({})
#     }
#     logger.info(f"DB counts: {counts}")



# async def initialize_db():
#     if SEED_DEMO:
#         await clear_database()
#         await populate_users()
#         await populate_conversations()
#         await populate_journals()
#         await verify_data()
#     await initialize_faiss_store()


# # ---------------------
# # Run (local only)
# # ---------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=PORT, proxy_headers=True, timeout_keep_alive=70)







#     uvicorn.run(app, host="0.0.0.0", port=PORT, proxy_headers=True, timeout_keep_alive=70)
