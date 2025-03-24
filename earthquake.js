document.addEventListener("DOMContentLoaded", function () {
    const predictBtn = document.getElementById("predictBtn");
    const damageText = document.getElementById("damage");
    const populationText = document.getElementById("population");
    const mapDiv = document.getElementById("map");
    let map;
    let markers = [];

    async function fetchPredictions() {
        const magnitude = localStorage.getItem("magnitude") || 5.5;
        const population = localStorage.getItem("population") || 100000;
        const infrastructure = localStorage.getItem("infrastructureCost") || 500;

        const requestData = {
            disaster_type: "Earthquake",
            magnitude: parseFloat(magnitude),
            population: parseInt(population),
            infrastructure: parseFloat(infrastructure)
        };

        try {
            const response = await fetch("https://back2-production-e0f7.up.railway.app", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            damageText.innerText = `$${result.total_damage_estimation}`;
            populationText.innerText = `${result.estimated_affected_population} people`;

            plotBuildingsOnMap(result.building_wise_impact);
            generateDamageChart(result.building_wise_impact);
        } catch (error) {
            console.error("Error fetching predictions:", error);
        }
    }

    async function plotBuildingsOnMap(buildings) {
        map = new google.maps.Map(mapDiv, {
            center: { lat: 19.076, lng: 72.8777 }, // Mumbai
            zoom: 12
        });

        markers.forEach(marker => marker.setMap(null)); // Clear existing markers
        markers = [];

        for (const building of buildings) {
            await new Promise(resolve => setTimeout(resolve, 500)); // Delay for API limit

            const buildingName = building[0][1]; // Get Building Name
            const damageValue = building[1][1]; // Get Damage Value
            const riskLevel = getRiskLevel(damageValue); // Determine Risk Level

            const geocodeURL = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(buildingName)}, Mumbai, India&key=YOUR_GOOGLE_MAPS_API_KEY`;

            try {
                const geoResponse = await fetch(geocodeURL);
                const geoData = await geoResponse.json();

                if (geoData.status === "OK") {
                    const location = geoData.results[0].geometry.location;
                    addMarker(location, buildingName, damageValue, riskLevel);
                } else {
                    console.warn(`Geocoding failed for ${buildingName}:`, geoData.status);
                }
            } catch (error) {
                console.error(`Geocoding failed for ${buildingName}:`, error);
            }
        }
    }

    function addMarker(location, buildingName, damageValue, riskLevel) {
        const markerIcon = {
            red: "http://maps.google.com/mapfiles/ms/icons/red-dot.png", // High Risk
            orange: "http://maps.google.com/mapfiles/ms/icons/orange-dot.png", // Medium Risk
            green: "http://maps.google.com/mapfiles/ms/icons/green-dot.png" // Low Risk
        };

        const marker = new google.maps.Marker({
            position: location,
            map: map,
            title: `${buildingName} - Damage: ${damageValue}`,
            icon: {
                url: markerIcon[riskLevel],
                scaledSize: new google.maps.Size(40, 40)
            }
        });

        const infoWindow = new google.maps.InfoWindow({
            content: `<b>${buildingName}</b><br>Estimated Damage: $${damageValue}<br>Risk Level: <span style="color:${riskLevel};">${riskLevel.toUpperCase()}</span>`
        });

        marker.addListener("click", () => infoWindow.open(map, marker));

        markers.push(marker);
    }

    function getRiskLevel(damageValue) {
        if (damageValue > 90) return "red"; // High Risk
        if (damageValue > 60 & damageValue < 90) return "orange"; // Medium Risk
        return "green"; // Low Risk
    }

    function generateDamageChart(buildings) {
        const ctx = document.getElementById("damageChart").getContext("2d");
        const labels = buildings.map(building => building[0][1]);
        const damageValues = buildings.map(building => building[1][1]);

        new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Estimated Damage",
                    data: damageValues,
                    backgroundColor: "rgba(255, 99, 132, 0.6)"
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } }
            }
        });
    }

    predictBtn.addEventListener("click", fetchPredictions);
});
