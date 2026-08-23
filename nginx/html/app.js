const form = document.getElementById("predict-form");
const resultDiv = document.getElementById("result");
const modelInfoP = document.getElementById("model-info");

const REQUIRED_FIELDS = [
    "gender", "study_time_hours", "attendance_percent", "sleep_hours",
    "parental_education", "internet_access", "extracurricular_activities",
    "part_time_job", "previous_grade",
];
const NUMERIC_FIELDS = ["study_time_hours", "attendance_percent", "sleep_hours", "previous_grade"];

const csvInput = document.getElementById("csv-input");
const batchBtn = document.getElementById("batch-predict-btn");
const batchResultDiv = document.getElementById("batch-result");

let parsedRows = [];

const FIELD_LABELS = {
    gender: "Género",
    study_time_hours: "Horas de estudio",
    attendance_percent: "Asistencia (%)",
    sleep_hours: "Horas de sueño",
    parental_education: "Educación de los padres",
    internet_access: "Acceso a internet",
    extracurricular_activities: "Actividades extracurriculares",
    part_time_job: "Trabajo part-time",
    previous_grade: "Nota anterior",
};

function formatValidationErrors(detail, rowLabels = null) {
    // detail viene del 422 de FastAPI/Pydantic: lista de {loc, msg, input, ...}
    if (!Array.isArray(detail)) return [String(detail)];

    return detail.map((err) => {
        const loc = err.loc || [];
        const isBatchRow = loc[1] === "students" && loc.length === 4;

        const field = isBatchRow ? loc[3] : loc[loc.length - 1];
        const fieldLabel = FIELD_LABELS[field] || field;
        const received = err.input !== undefined ? ` (recibido: "${err.input}")` : "";

        if (isBatchRow) {
            const rowIndex = loc[2];
            const rowLabel = rowLabels?.[rowIndex] ?? `fila ${rowIndex + 1}`;
            return `Student_ID: ${rowLabel} - "${fieldLabel}": ${err.msg}${received}`;
        }

        return `"${fieldLabel}": ${err.msg}${received}`;
    });
}

function renderErrors(container, messages) {
    container.className = "error";
    container.innerHTML = `<ul>${messages.map((m) => `<li>${m}</li>`).join("")}</ul>`;
}

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
            const messages = formatValidationErrors(data.detail);
            renderErrors(resultDiv, messages);
            return;
        }

        resultDiv.className = "ok";
        resultDiv.textContent = `Nota predicha: ${data.final_grade} (modelo v${data.model_version})`;
    } catch (e) {
        resultDiv.className = "error";
        resultDiv.textContent = "No se pudo conectar con la API.";
    }
});

function parseCsv(text) {
    const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
    if (lines.length < 2) throw new Error("El CSV no tiene filas de datos.");

    const headers = lines[0].split(",").map((h) => h.trim());
    const missing = REQUIRED_FIELDS.filter((f) => !headers.includes(f));
    if (missing.length > 0) {
        throw new Error(`Faltan columnas en el CSV: ${missing.join(", ")}`);
    }

    return lines.slice(1).map((line, idx) => {
        const values = line.split(",").map((v) => v.trim());
        const row = {};
        headers.forEach((h, i) => { row[h] = values[i]; });

        const student = {};
        for (const field of REQUIRED_FIELDS) {
            student[field] = NUMERIC_FIELDS.includes(field) ? parseFloat(row[field]) : row[field];
        }
        return { label: row.student_id ?? `fila ${idx + 1}`, student };
    });
}

csvInput.addEventListener("change", () => {
    const file = csvInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
        try {
            parsedRows = parseCsv(reader.result);
            batchResultDiv.textContent = `${parsedRows.length} filas listas para predecir.`;
            batchResultDiv.className = "";
            batchBtn.disabled = false;
        } catch (e) {
            parsedRows = [];
            batchResultDiv.textContent = `Error al leer el CSV: ${e.message}`;
            batchResultDiv.className = "error";
            batchBtn.disabled = true;
        }
    };
    reader.readAsText(file);
});

batchBtn.addEventListener("click", async () => {
    if (parsedRows.length === 0) return;

    batchResultDiv.className = "";
    batchResultDiv.textContent = "Prediciendo batch...";

    try {
        const res = await fetch("/predict/batch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ students: parsedRows.map((r) => r.student) }),
        });
        const data = await res.json();

        if (!res.ok) {
            const rowLabels = parsedRows.map((r) => r.label);
            const messages = formatValidationErrors(data.detail, rowLabels);
            renderErrors(batchResultDiv, messages);
            return;
        }

        const rowsHtml = parsedRows
            .map((r, i) => `<tr><td>${r.label}</td><td>${data.predictions[i]}</td></tr>`)
            .join("");
        batchResultDiv.className = "ok";
        batchResultDiv.innerHTML = `
            <p>Modelo v${data.model_version}</p>
            <table>
                <thead><tr><th>Estudiante</th><th>Predicción</th></tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        `;
    } catch (e) {
        batchResultDiv.className = "error";
        batchResultDiv.textContent = "No se pudo conectar con la API.";
    }
});

loadModelInfo();