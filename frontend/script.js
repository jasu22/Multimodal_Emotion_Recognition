const predictButton = document.getElementById("predictButton");

const emotions = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
];


/* =========================
   EMOTION EMOJIS
========================= */

const emotionEmojis = {
    angry: "😡",
    disgust: "🤢",
    fearful: "😨",
    happy: "😊",
    neutral: "😐",
    sad: "😢",
    surprised: "😮"
};


/* =========================
   FILE NAME DISPLAY
========================= */

const audioInput = document.getElementById("audioInput");

const imageInput = document.getElementById("imageInput");


audioInput.addEventListener("change", function () {

    const file = audioInput.files[0];

    if (!file) {
        return;
    }

    document.getElementById("audioFileName").innerHTML = `

        <i class="fa-solid fa-music"></i>

        <div>

            <strong>${file.name}</strong>

            <small>
                ${(file.size / 1024 / 1024).toFixed(2)} MB
            </small>

        </div>
    `;
});


imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    document.getElementById("imageFileName").innerHTML = `

        <i class="fa-solid fa-image"></i>

        <div>

            <strong>${file.name}</strong>

            <small>
                ${(file.size / 1024 / 1024).toFixed(2)} MB
            </small>

        </div>
    `;


    /* Show image preview */

    const imageURL = URL.createObjectURL(file);

    const facePreview =
        document.getElementById("facePreview");

    facePreview.src = imageURL;
});


/* =========================
   TEXT CHARACTER COUNTER
========================= */

const textInput =
    document.getElementById("textInput");

const charCount =
    document.getElementById("charCount");


textInput.addEventListener("input", function () {

    charCount.textContent =
        textInput.value.length;

});


/* =========================
   CREATE PROBABILITY ROW
========================= */

function createProbabilityRow(
    emotion,
    probability,
    type
) {

    const percentage =
        probability * 100;


    const row =
        document.createElement("div");

    row.className =
        "probability-row";


    const name =
        document.createElement("span");

    name.textContent =
        emotion.charAt(0).toUpperCase()
        + emotion.slice(1);


    const barContainer =
        document.createElement("div");

    barContainer.className =
        "probability-bar-container";


    const bar =
        document.createElement("div");

    bar.className =
        "probability-bar";


    if (type === "audio") {

        bar.style.background =
            "#2478ed";

    }

    else if (type === "text") {

        bar.style.background =
            "#19a96d";

    }

    else if (type === "face") {

        bar.style.background =
            "#6045dc";

    }


    bar.style.width = "0%";


    barContainer.appendChild(bar);


    const value =
        document.createElement("span");

    value.className =
        "probability-value";

    value.textContent =
        percentage.toFixed(2);


    row.appendChild(name);

    row.appendChild(barContainer);

    row.appendChild(value);


    /* Animate bar */

    setTimeout(function () {

        bar.style.width =
            percentage + "%";

    }, 100);


    return row;
}


/* =========================
   DISPLAY PROBABILITIES
========================= */

function displayProbabilities(
    containerId,
    probabilities,
    type
) {

    const container =
        document.getElementById(containerId);

    container.innerHTML = "";


    emotions.forEach(function (emotion) {

        const probability =
            probabilities[emotion] || 0;


        const row =
            createProbabilityRow(
                emotion,
                probability,
                type
            );


        container.appendChild(row);

    });
}


/* =========================
   ANALYZE EMOTION
========================= */

predictButton.addEventListener(
    "click",
    async function () {


        const text =
            textInput.value.trim();


        const audioFile =
            audioInput.files[0];


        const imageFile =
            imageInput.files[0];


        /* Validate input */

        if (
            !text ||
            !audioFile ||
            !imageFile
        ) {

            alert(
                "Please enter text and upload both audio and image."
            );

            return;
        }


        /* Create form */

        const formData =
            new FormData();


        formData.append(
            "audio",
            audioFile
        );


        formData.append(
            "image",
            imageFile
        );


        /* Loading */

        document
            .getElementById("loading")
            .classList
            .remove("hidden");


        document
            .getElementById("result")
            .classList
            .add("hidden");


        predictButton.disabled = true;

        predictButton.innerHTML = `

            <i class="fa-solid fa-spinner fa-spin"></i>

            Analyzing...

        `;


        try {


            /* =========================
               SEND REQUEST TO FASTAPI
            ========================= */

            const response =
                await fetch(

                    `http://127.0.0.1:8000/predict-multimodal?text=${encodeURIComponent(text)}`,

                    {

                        method: "POST",

                        body: formData

                    }

                );


            if (!response.ok) {

                const errorData =
                    await response.json();

                console.log(
                    "FastAPI Error:",
                    errorData
                );


                alert(
                    JSON.stringify(errorData)
                );


                return;
            }


            /* =========================
               GET API RESPONSE
            ========================= */

            const data =
                await response.json();


            console.log(
                "Multimodal Result:",
                data
            );


            /* =========================
               AUDIO
            ========================= */

            document
                .getElementById("audioEmotion")
                .textContent =
                data.audio_emotion;


            displayProbabilities(

                "audioProbabilities",

                data.audio_probabilities,

                "audio"

            );


            /* =========================
               TEXT
            ========================= */

            document
                .getElementById("textEmotion")
                .textContent =
                data.text_emotion;


            document
                .getElementById("textPreview")
                .textContent =
                `"${data.text}"`;


            displayProbabilities(

                "textProbabilities",

                data.text_probabilities,

                "text"

            );


            /* =========================
               FACE
            ========================= */

            document
                .getElementById("faceEmotion")
                .textContent =
                data.face_emotion;


            displayProbabilities(

                "faceProbabilities",

                data.face_probabilities,

                "face"

            );


            /* =========================
               LATE FUSION
            ========================= */

            const lateEmotion =
                data.final_emotion;


            const lateConfidence =
                data.fused_probabilities[
                    lateEmotion
                ] * 100;


            document
                .getElementById(
                    "lateFusionEmotion"
                )
                .textContent =
                lateEmotion;


            /* =========================
               FINAL EMOTION
            ========================= */

            document
                .getElementById(
                    "finalEmotion"
                )
                .textContent =
                lateEmotion;


            document
                .getElementById(
                    "finalConfidence"
                )
                .textContent =
                lateConfidence.toFixed(2);


            document
                .getElementById(
                    "finalEmoji"
                )
                .textContent =
                emotionEmojis[
                    lateEmotion
                ] || "😊";


            document
                .getElementById(
                    "finalProgress"
                )
                .style.width =
                lateConfidence + "%";


            /* =========================
               FINAL FUSION PROBABILITIES
            ========================= */

            console.log(
                "Fused probabilities:",
                data.fused_probabilities
            );


            /* =========================
               EARLY FUSION
            ========================= */

            document
                .getElementById(
                    "earlyFusionEmotion"
                )
                .textContent =
                "Evaluation Model";


            /* =========================
               SHOW RESULTS
            ========================= */

            document
                .getElementById("result")
                .classList
                .remove("hidden");


        }

        catch (error) {

            console.error(error);


            alert(

                "Unable to connect to the FastAPI server.\n\n" +

                "Make sure the server is running."

            );

        }


        finally {

            document
                .getElementById("loading")
                .classList
                .add("hidden");


            predictButton.disabled =
                false;


            predictButton.innerHTML = `

                <i class="fa-solid fa-play"></i>

                Analyze Emotion

            `;

        }

    }
);