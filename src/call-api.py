from gradio_client import Client

# client = Client("http://127.0.0.1:7860/")
client = Client("pixelciosa/estimador_alquileres_oficinas_caba")
result = client.predict(
	param_0=1,
	param_1=4,
	param_2=600,
	param_3="Catalinas",
	param_4=False,
	param_5=True,
	api_name="/prediccion"
)
print(result)