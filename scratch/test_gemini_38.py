import time
import google.protobuf.json_format as jf
orig_m2d = jf.MessageToDict
def patched_m2d(*args, **kwargs):
    kwargs.pop("float_precision", None)
    return orig_m2d(*args, **kwargs)
jf.MessageToDict = patched_m2d

import google.generativeai as genai

key = "AQ.Ab8RN6Js3tnsQphiIexiDXbM9RRUssT2ACcG9XHVr638djpA6A"
genai.configure(api_key=key)

print("Waiting 7 seconds for rate limit window...")
time.sleep(7)

m = genai.GenerativeModel("gemini-3.8-flash")
r = m.generate_content("What is the ratio decidendi in simple terms? Answer in 1 short sentence.")
print("Response from gemini-3.8-flash:")
print(r.text.strip())

