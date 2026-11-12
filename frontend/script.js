
"use strict";

// ==========================================
// CUSTOMER CHURN PREDICTION SYSTEM
// Frontend JavaScript
// ==========================================


// 1. Select HTML elements

const form = document.getElementById("predictionForm");

const predictButton = document.getElementById("predictButton");
const sampleButton = document.getElementById("sampleButton");
const refreshButton = document.getElementById("refreshButton");

const formMessage = document.getElementById("formMessage");

const emptyResult = document.getElementById("emptyResult");
const predictionResult = document.getElementById("predictionResult");

const predictionLabel = document.getElementById("predictionLabel");
const probabilityValue = document.getElementById("probabilityValue");
const probabilityFill = document.getElementById("probabilityFill");
const probabilityMeter = document.getElementById("probabilityMeter");

const riskBadge = document.getElementById("riskBadge");
const resultDescription = document.getElementById("resultDescription");

const historyBody = document.getElementById("historyBody");
const historyCount = document.getElementById("historyCount");


// 2. Display success and error messages

function showMessage(message, kind = "") {

    formMessage.textContent = message;
    formMessage.dataset.kind = kind;

}


// 3. Send HTTP requests to FastAPI

async function apiRequest(url, options = {}) {

    const response = await fetch(url, options);

    let data;

    try {

        data = await response.json();

    } catch {

        throw new Error(
            `Invalid server response (HTTP ${response.status}).`
        );

    }

    if (!response.ok) {

        let message = `Request failed (HTTP ${response.status}).`;

        if (Array.isArray(data.detail)) {

            message = data.detail.map((item) => {

                const field = item.loc?.at(-1) ?? "Input";

                return `${field}: ${item.msg}`;

            }).join("; ");

        } else if (typeof data.detail === "string") {

            message = data.detail;

        }

        throw new Error(message);

    }

    return data;

}


// 4. Convert churn score to percentage

function toPercent(probability) {

    const number = Number(probability);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return (number * 100).toFixed(2) + "%";

}


// 5. Display customer prediction results

function displayPrediction(data) {

    const probability = Number(data.probability);

    if (
        !Number.isFinite(probability) ||
        probability < 0 ||
        probability > 1
    ) {
        throw new Error("Invalid prediction score received.");
    }

    const risk = data.risk_level;

    if (!["Low", "Medium", "High"].includes(risk)) {
        throw new Error("Invalid risk level received.");
    }

    // Display predicted class
    predictionLabel.textContent = data.prediction;

    // Display churn score
    probabilityValue.textContent = toPercent(probability);

    // Display risk badge
    riskBadge.textContent = risk + " Risk";

    riskBadge.className =
        "risk-badge " + risk.toLowerCase();

    // Update probability bar
    const percentage = probability * 100;

    probabilityFill.style.width = percentage + "%";

    probabilityMeter.setAttribute(
        "aria-valuenow",
        percentage.toFixed(2)
    );

    // Risk-based colors and explanation
    if (risk === "High") {

        probabilityFill.style.background = "#dc2626";

        resultDescription.textContent =
            "This customer has a high churn score. " +
            "Consider reviewing their retention needs.";

    } else if (risk === "Medium") {

        probabilityFill.style.background = "#d97706";

        resultDescription.textContent =
            "This customer has a moderate churn score. " +
            "Monitoring their engagement may be helpful.";

    } else {

        probabilityFill.style.background = "#16a34a";

        resultDescription.textContent =
            "This customer has a low churn score " +
            "under the current model.";

    }

    // Show prediction results
    emptyResult.hidden = true;

    predictionResult.hidden = false;

}


// 6. Handle customer prediction form

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    // Check form validation
    if (!form.reportValidity()) {
        return;
    }

    // Get customer information
    const values = new FormData(form);

    const customer = {

        tenure: Number(values.get("tenure")),

        monthly_charges: Number(
            values.get("monthly_charges")
        ),

        total_charges: Number(
            values.get("total_charges")
        ),

        contract: values.get("contract"),

        internet_service: values.get("internet_service")

    };

    // Disable button during prediction
    predictButton.disabled = true;

    predictButton.textContent = "Predicting...";

    showMessage("Processing customer information...");

    try {

        // Send customer data to FastAPI
        const result = await apiRequest("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(customer)

        });

        // Display prediction
        displayPrediction(result);

        showMessage(
            "Prediction completed and saved successfully.",
            "success"
        );

        // Update prediction history
        await loadPredictionHistory();

    } catch (error) {

        console.error("Prediction error:", error);

        showMessage(
            error.message,
            "error"
        );

    } finally {

        // Enable button again
        predictButton.disabled = false;

        predictButton.textContent =
            "Predict Customer Churn";

    }

});


// 7. Load sample customer information

sampleButton.addEventListener("click", function () {

    document.getElementById("tenure").value = 12;

    document.getElementById("monthly_charges").value = 75.5;

    document.getElementById("total_charges").value = 906;

    document.getElementById("contract").value =
        "Month-to-month";

    document.getElementById("internet_service").value =
        "Fiber optic";

    showMessage(
        "Sample customer information loaded."
    );

});


// 8. Clear customer information

form.addEventListener("reset", function () {

    showMessage("");

    predictionResult.hidden = true;

    emptyResult.hidden = false;

});


// 9. Load machine learning model information

async function loadModelInformation() {

    try {

        const info = await apiRequest("/model/info");

        document.getElementById("modelName").textContent =
            info.model_name ?? "Unknown";

        document.getElementById("modelAccuracy").textContent =
            toPercent(info.accuracy);

        document.getElementById("modelRecall").textContent =
            toPercent(info.recall);

        document.getElementById("modelF1").textContent =
            toPercent(info.f1_score);

    } catch (error) {

        console.error(
            "Model information error:",
            error
        );

        document.getElementById("modelName").textContent =
            "Unavailable";

    }

}


// 10. Format SQLite timestamps

function formatTimestamp(timestamp) {

    if (!timestamp) {
        return "—";
    }

    // SQLite CURRENT_TIMESTAMP uses UTC
    const iso = timestamp.replace(" ", "T") + "Z";

    const date = new Date(iso);

    if (Number.isNaN(date.getTime())) {
        return timestamp;
    }

    return date.toLocaleString();

}


// 11. Create prediction history table rows

function createHistoryRow(record) {

    const row = document.createElement("tr");

    const values = [

        formatTimestamp(record.created_at),

        record.tenure,

        record.contract,

        toPercent(record.probability),

        record.risk_level,

        record.prediction

    ];

    values.forEach(function (value) {

        const cell = document.createElement("td");

        cell.textContent = value ?? "—";

        row.appendChild(cell);

    });

    return row;

}


// 12. Retrieve saved predictions from SQLite

async function loadPredictionHistory() {

    try {

        const data = await apiRequest(
            "/predictions?limit=10&offset=0"
        );

        // Clear previous table rows
        historyBody.replaceChildren();

        const records = data.predictions ?? [];

        if (records.length === 0) {

            const row = document.createElement("tr");

            const cell = document.createElement("td");

            cell.colSpan = 6;

            cell.textContent =
                "No predictions saved yet.";

            row.appendChild(cell);

            historyBody.appendChild(row);

        } else {

            records.forEach(function (record) {

                historyBody.appendChild(
                    createHistoryRow(record)
                );

            });

        }

        historyCount.textContent =
            `Showing ${records.length} of ${data.total} saved predictions`;

    } catch (error) {

        console.error(
            "Prediction history error:",
            error
        );

        historyBody.replaceChildren();

        const row = document.createElement("tr");

        const cell = document.createElement("td");

        cell.colSpan = 6;

        cell.textContent =
            "Unable to load prediction history.";

        row.appendChild(cell);

        historyBody.appendChild(row);

        historyCount.textContent =
            "History unavailable";

    }

}


// 13. Refresh prediction history

refreshButton.addEventListener(
    "click",
    loadPredictionHistory
);


// 14. Initialize frontend

loadModelInformation();

loadPredictionHistory();
