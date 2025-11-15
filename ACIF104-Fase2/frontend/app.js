const apiUrl = "http://localhost:8000/predict";

document.getElementById("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const inputs = document.querySelectorAll("#features-container input");
    const features = {};

    inputs.forEach(input => {
        if (input.name && input.value !== "") {
            features[input.name] = parseFloat(input.value);
        }
    });

    const payload = { features };

    try {
        const resp = await fetch(apiUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!resp.ok) {
            const err = await resp.json();
            document.getElementById("resultado").innerText = "Error: " + err.detail;
            return;
        }

        const data = await resp.json();
        document.getElementById("resultado").innerText =
            "Predicción de demanda: " + data.prediction.toFixed(2);

    } catch (error) {
        document.getElementById("resultado").innerText =
            "Error al conectar con la API: " + error;
    }
});
