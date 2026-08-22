const form = document.getElementById("predict-form");
const resultDiv = document.getElementById("result");
const modelInfoP = document.getElementById("model-info");

async function loadModelInfo() {
    try {
        const res = await fetch("/model-info");
        if (!res.ok) {
            modelInfoP.textContent = "Modelo: no hay champion registrado todavía.";
            return;
        }
        const data = await res.json();
        modelInfoP.textContent = `Modelo: ${data.name} v${data.version} (f1_weighted: ${data.f1_weighted?.toFixed(3) ?? "N/A"})`;
    } catch (e) {
        modelInfoP.textContent = "No se pudo consultar /model-info.";
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    resultDiv.className = "";
    resultDiv.textContent = "Prediciendo...";

    const formData = new FormData(form);
    const payload = {
        gender: formData.get("gender"),
        study_time_hours: parseFloat(formData.get("study_time_hours")),
        attendance_percent: parseFloat(formData.get("attendance_percent")),
        sleep_hours: parseFloat(formData.get("sleep_hours")),
        parental_education: formData.get("parental_education"),
        internet_access: formData.get("internet_access"),
        extracurricular_activities: formData.get("extracurricular_activities"),
        part_time_job: formData.get("part_time_job"),
        previous_grade: parseFloat(formData.get("previous_grade")),
    };

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        if (!res.ok) {
            resultDiv.className = "error";
            resultDiv.textContent = `Error: ${JSON.stringify(data.detail)}`;
            return;
        }

        resultDiv.className = "ok";
        resultDiv.textContent = `Nota predicha: ${data.final_grade} (modelo v${data.model_version})`;
    } catch (e) {
        resultDiv.className = "error";
        resultDiv.textContent = "No se pudo conectar con la API.";
    }
});

loadModelInfo();