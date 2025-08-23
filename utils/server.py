import phoenix as px

session = px.launch_app()
print(f"Phoenix launched! URL: {session.url}")

# uv run -m phoenix.server.main serve