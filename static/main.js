document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("upload-form").addEventListener("submit", function (event) {
        event.preventDefault();
        analyzeFile();
    });

    document.getElementById("analyze-text").addEventListener("click", function () {
        analyzeText();
    });
});

function analyzeFile() {
    var form = new FormData(document.getElementById("upload-form"));
    sendAnalysisRequest(form, true); // Add 'true' to indicate file analysis
}

function analyzeText() {
    var text = document.getElementById("text-input").value;
    var form = new FormData();
    form.append("text", text);
    sendAnalysisRequest(form, false); // Add 'false' to indicate text analysis
}

function sendAnalysisRequest(formData, isFile) {
    document.getElementById("pie-chart-container").style.display = "none";
    document.getElementById("sentiment-counts").style.display = "none";

    fetch("/analyze", {
        method: "POST",
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            document.getElementById("pie-chart").src = "data:image/png;base64," + data.image;
            document.getElementById("pie-chart-container").style.display = "block";

            document.getElementById("positive-count").textContent = "Positive: " + data.sentiment_counts.positive;
            document.getElementById("neutral-count").textContent = "Neutral: " + data.sentiment_counts.neutral;
            document.getElementById("negative-count").textContent = "Negative: " + data.sentiment_counts.negative;
            document.getElementById("sentiment-counts").style.display = "block";
        })
        .catch(error => console.error("Error:", error));
}