import app
from flask import Flask

ctx = app.app.test_request_context('/health')
ctx.push()
resp = app.health()
print(resp.status_code)
print(resp.get_data(as_text=True))
ctx.pop()
