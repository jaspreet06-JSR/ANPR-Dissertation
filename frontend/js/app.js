const API_ENDPOINT = "/api/detection/image";


/* =====================================================
   ELEMENTS
===================================================== */

const imageInput =
    document.getElementById("imageInput");

const selectImageButton =
    document.getElementById("selectImageButton");

const dropZone =
    document.getElementById("dropZone");

const fileName =
    document.getElementById("fileName");

const analyzeButton =
    document.getElementById("analyzeButton");

const resultCanvas =
    document.getElementById("resultCanvas");

const imagePlaceholder =
    document.getElementById("imagePlaceholder");

const vehicleResults =
    document.getElementById("vehicleResults");

const loadingOverlay =
    document.getElementById("loadingOverlay");

const errorToast =
    document.getElementById("errorToast");

const vehicleCount =
    document.getElementById("vehicleCount");

const plateCount =
    document.getElementById("plateCount");

const ocrCount =
    document.getElementById("ocrCount");

const processingTime =
    document.getElementById("processingTime");

const vehiclePipeline =
    document.getElementById("vehiclePipeline");


let selectedFile = null;


/* =====================================================
   FILE SELECTION
===================================================== */

selectImageButton.addEventListener(
    "click",
    function (event) {

        event.stopPropagation();

        imageInput.click();

    }
);


dropZone.addEventListener(
    "click",
    function () {

        imageInput.click();

    }
);


imageInput.addEventListener(
    "change",
    function () {

        if (this.files.length === 0) {
            return;
        }

        handleSelectedFile(
            this.files[0]
        );

    }
);


/* =====================================================
   DRAG & DROP
===================================================== */

dropZone.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropZone.style.borderColor =
            "#38bdf8";

    }
);


dropZone.addEventListener(
    "dragleave",
    function () {

        dropZone.style.borderColor =
            "";

    }
);


dropZone.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropZone.style.borderColor =
            "";

        const files =
            event.dataTransfer.files;

        if (files.length > 0) {

            handleSelectedFile(
                files[0]
            );

        }

    }
);


/* =====================================================
   HANDLE FILE
===================================================== */

function handleSelectedFile(file) {

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {

        showError(
            "Please select a JPG, PNG or WEBP image."
        );

        return;
    }


    selectedFile = file;

    fileName.textContent =
        file.name;

    analyzeButton.disabled =
        false;


    // Preview image immediately

    const imageURL =
        URL.createObjectURL(file);

    drawImage(
        imageURL,
        []
    );

}


/* =====================================================
   ANALYZE IMAGE
===================================================== */

analyzeButton.addEventListener(
    "click",
    analyzeImage
);


async function analyzeImage() {

    if (!selectedFile) {

        showError(
            "Please select an image first."
        );

        return;
    }


    setLoading(true);

    analyzeButton.disabled =
        true;


    vehiclePipeline.classList.add(
        "active"
    );


    const formData =
        new FormData();

    formData.append(
        "file",
        selectedFile
    );


    try {

        const response =
            await fetch(
                API_ENDPOINT,
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Image analysis failed."
            );

        }


        displayResults(
            data
        );

        loadHistory();


    } catch (error) {

        console.error(
            error
        );

        showError(
            error.message
        );

    } finally {

        setLoading(false);

        analyzeButton.disabled =
            false;

    }
}


/* =====================================================
   DISPLAY RESULTS
===================================================== */

function displayResults(data) {

    console.log(
        "ANPR Result:",
        data
    );


    /*
     * Update statistics
     */

    vehicleCount.textContent =
        data.summary?.vehicles_detected ?? 0;

    plateCount.textContent =
        data.summary?.plates_detected ?? 0;

    ocrCount.textContent =
        data.summary?.ocr_results ?? 0;


    const time =
        data.processing?.time_ms;

    processingTime.textContent =
        time !== undefined
            ? `${time} ms`
            : "--";


    /*
     * Display vehicle cards
     */

    vehicleResults.innerHTML =
        "";


    if (
        !data.vehicles ||
        data.vehicles.length === 0
    ) {

        vehicleResults.innerHTML =
            `<div class="empty-result">
                No vehicles detected.
            </div>`;

    } else {

        data.vehicles.forEach(
            function (vehicle) {

                const card =
                    document.createElement(
                        "div"
                    );

                card.className =
                    "vehicle-card";


                const confidence =
                    (
                        vehicle.confidence *
                        100
                    ).toFixed(1);


                const bbox =
                    vehicle.bbox;


                card.innerHTML = `

                    <div class="vehicle-card-header">

                        <span class="vehicle-type">
                            ${vehicle.vehicle_type}
                        </span>

                        <span class="confidence">
                            ${confidence}%
                        </span>

                    </div>


                    <div class="vehicle-info">

                        <div class="info-item">

                            <span>
                                VEHICLE ID
                            </span>

                            <strong>
                                #${vehicle.vehicle_id}
                            </strong>

                        </div>


                        <div class="info-item">

                            <span>
                                CONFIDENCE
                            </span>

                            <strong>
                                ${confidence}%
                            </strong>

                        </div>


                        <div class="info-item">

                            <span>
                                X1 / Y1
                            </span>

                            <strong>
                                ${bbox.x1}, ${bbox.y1}
                            </strong>

                        </div>


                        <div class="info-item">

                            <span>
                                X2 / Y2
                            </span>

                            <strong>
                                ${bbox.x2}, ${bbox.y2}
                            </strong>

                        </div>

                    </div>


                    <div class="plate-status">

                        License Plate:
                        <strong>
                            Not detected yet
                        </strong>

                    </div>
                `;


                vehicleResults.appendChild(
                    card
                );

            }
        );

    }


    /*
     * Draw bounding boxes
     */

    if (data.files?.annotated_image) {

        const annotatedURL =
            data.files.annotated_image;

        drawImage(
            annotatedURL,
            []
        );

    } else {

        const imageURL =
            URL.createObjectURL(
                selectedFile
            );


        drawImage(
            imageURL,
            data.vehicles || []
        );
    }

    vehiclePipeline.classList.remove(
        "active"
    );

    vehiclePipeline.classList.add(
        "completed"
    );

}


/* =====================================================
   DRAW IMAGE + BOUNDING BOXES
===================================================== */

function drawImage(
    imageURL,
    vehicles
) {

    const image =
        new Image();


    image.onload =
        function () {

            const maxWidth = 1000;
            const maxHeight = 600;

            let width =
                image.naturalWidth;

            let height =
                image.naturalHeight;


            const scale =
                Math.min(
                    maxWidth / width,
                    maxHeight / height,
                    1
                );


            width *= scale;
            height *= scale;


            resultCanvas.width =
                width;

            resultCanvas.height =
                height;


            const ctx =
                resultCanvas.getContext(
                    "2d"
                );


            ctx.clearRect(
                0,
                0,
                width,
                height
            );


            ctx.drawImage(
                image,
                0,
                0,
                width,
                height
            );


            /*
             * Draw YOLO bounding boxes
             */

            vehicles.forEach(
                function (vehicle) {

                    const box =
                        vehicle.bbox;


                    const scaleX =
                        width /
                        image.naturalWidth;

                    const scaleY =
                        height /
                        image.naturalHeight;


                    const x =
                        box.x1 *
                        scaleX;

                    const y =
                        box.y1 *
                        scaleY;

                    const boxWidth =
                        (box.x2 - box.x1) *
                        scaleX;

                    const boxHeight =
                        (box.y2 - box.y1) *
                        scaleY;


                    ctx.strokeStyle =
                        "#22c55e";

                    ctx.lineWidth =
                        3;


                    ctx.strokeRect(
                        x,
                        y,
                        boxWidth,
                        boxHeight
                    );


                    /*
                     * Label background
                     */

                    const label =
                        `${vehicle.vehicle_type} ` +
                        `${(
                            vehicle.confidence *
                            100
                        ).toFixed(1)}%`;


                    ctx.font =
                        "bold 14px Arial";


                    const textWidth =
                        ctx.measureText(
                            label
                        ).width;


                    ctx.fillStyle =
                        "#22c55e";


                    ctx.fillRect(
                        x,
                        Math.max(
                            0,
                            y - 24
                        ),
                        textWidth + 14,
                        24
                    );


                    ctx.fillStyle =
                        "#07110a";


                    ctx.fillText(
                        label,
                        x + 7,
                        Math.max(
                            17,
                            y - 7
                        )
                    );

                }
            );


            resultCanvas.style.display =
                "block";

            imagePlaceholder.style.display =
                "none";

        };


    image.src =
        imageURL;
}


/* =====================================================
   NAVIGATION
===================================================== */

const navItems =
    document.querySelectorAll(
        ".nav-item"
    );


const sections =
    document.querySelectorAll(
        ".page-section"
    );


navItems.forEach(
    function (button) {

        button.addEventListener(
            "click",
            function () {

                const target =
                    button.dataset.section;


                navItems.forEach(
                    item => {
                        item.classList.remove(
                            "active"
                        );
                    }
                );


                sections.forEach(
                    section => {
                        section.classList.remove(
                            "active-section"
                        );
                    }
                );


                button.classList.add(
                    "active"
                );


                const targetSection =
                    document.getElementById(
                        target
                    );


                if (targetSection) {

                    targetSection.classList.add(
                        "active-section"
                    );

                }

            }
        );

    }
);


/* =====================================================
   LOADING
===================================================== */

function setLoading(isLoading) {

    if (isLoading) {

        loadingOverlay.classList.add(
            "visible"
        );

    } else {

        loadingOverlay.classList.remove(
            "visible"
        );

    }
}


/* =====================================================
   ERROR
===================================================== */

function showError(message) {

    errorToast.textContent =
        message;

    errorToast.classList.add(
        "show"
    );


    setTimeout(
        function () {

            errorToast.classList.remove(
                "show"
            );

        },
        4000
    );

}

async function loadHistory() {

    const historyContent =
        document.getElementById(
            "historyContent"
        );

    try {

        const response =
            await fetch(
                "/api/results/"
            );

        const data =
            await response.json();


        if (
            !data.results ||
            data.results.length === 0
        ) {

            historyContent.innerHTML =
                `
                <div class="empty-result">
                    No previous results available.
                </div>
                `;

            return;
        }


        historyContent.innerHTML =
            "";


        data.results.forEach(
            function (result) {

                const item =
                    document.createElement(
                        "div"
                    );

                item.className =
                    "history-item";


                const vehicleCount =
                    result.summary
                    ?.vehicles_detected ?? 0;


                const plateCount =
                    result.summary
                    ?.plates_detected ?? 0;


                const time =
                    result.processing
                    ?.time_ms ?? 0;


                item.innerHTML = `

                    <div>

                        <strong>
                            ${result.image?.filename ?? "Unknown"}
                        </strong>

                        <div class="history-meta">

                            ${vehicleCount}
                            vehicle(s)

                            ·

                            ${plateCount}
                            plate(s)

                            ·

                            ${time} ms

                        </div>

                    </div>


                    <button
                        class="history-button"
                        onclick="viewHistoryResult(
                            '${result.result_id}'
                        )"
                    >
                        View
                    </button>
                `;


                historyContent.appendChild(
                    item
                );

            }
        );

    } catch (error) {

        console.error(
            "History error:",
            error
        );

        historyContent.innerHTML =
            `
            <div class="empty-result">
                Unable to load history.
            </div>
            `;
    }
}

async function viewHistoryResult(
    resultId
) {

    try {

        const response =
            await fetch(
                `/api/results/${resultId}`
            );


        const result =
            await response.json();


        if (!response.ok) {

            throw new Error(
                result.detail ||
                "Unable to load result."
            );
        }


        console.log(
            "Selected result:",
            result
        );


        alert(
            `Result ID: ${result.result_id}\n` +
            `Vehicles: ${
                result.summary?.vehicles_detected ?? 0
            }\n` +
            `Plates: ${
                result.summary?.plates_detected ?? 0
            }`
        );

    } catch (error) {

        showError(
            error.message
        );

    }
}