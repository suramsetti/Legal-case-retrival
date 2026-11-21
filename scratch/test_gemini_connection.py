import os
import google.protobuf.json_format as jf
orig_m2d = jf.MessageToDict
def patched_m2d(*args, **kwargs):
    kwargs.pop("float_precision", None)
    return orig_m2d(*args, **kwargs)
jf.MessageToDict = patched_m2d

import google.generativeai as genai

key = "AQ.Ab8RN6Js3tnsQphiIexiDXbM9RRUssT2ACcG9XHVr638djpA6A"
genai.configure(api_key=key)

try:
    print("Testing models listing...")
    models = list(genai.list_models())
    gemini_models = [m.name for m in models if "gemini" in m.name.lower()]
    print("Available Gemini models:", gemini_models[:5])
    
    candidates = ['gemini-flash-latest', 'gemini-2.5-pro', 'gemini-3.8-flash', 'gemini-2.0-flash']
    for m_name in candidates:
        try:
            print(f"Trying {m_name}...")
            m = genai.GenerativeModel(m_name)
            r = m.generate_content('{"status": "ok"}')
            print(f"--> SUCCESS with {m_name}! Response: {r.text.strip()[:100]}")
            break
        except Exception as err:
            print(f"--> FAILED with {m_name}: {err}")
except Exception as e:
    print("Connection FAILED:", type(e).__name__, str(e))
