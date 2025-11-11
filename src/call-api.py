from gradio_client import Client

client = Client("http://127.0.0.1:7869/")
result = client.predict(
	param_0=5,
	param_1=4,
	param_2=600,
	param_3="Catalinas",
	api_name="/prediccion"
)
print(result)