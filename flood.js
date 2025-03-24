document.addEventListener("DOMContentLoaded", function () {
    const predictBtn = document.getElementById("predictBtn");
    const damageText = document.getElementById("damage");
    const populationText = document.getElementById("population");
    const campsText = document.getElementById("camps");
    const boatsText = document.getElementById("boats");
    const suppliesText = document.getElementById("supplies");
    const cost = document.getElementById("Cost")
    const mapDiv = document.getElementById("map");
    let map;
    let floodZones = [];

    async function fetchPredictions() {
        const magnitude = localStorage.getItem("magnitude") || 6.0;
        const population = localStorage.getItem("population") || 100000;
        const infrastructure = localStorage.getItem("infrastructureCost") || 500;

        const requestData = {
            disaster_type: "Flood",
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

            damageText.innerText = `${result.total_damage_estimation}`*87000;
            populationText.innerText = `${result.estimated_affected_population}0 people`;

            // Estimate relief resources
            const camps = Math.ceil(result.estimated_affected_population *10/12);
            const boats = Math.ceil(result.estimated_affected_population );
            const supplies = result.estimated_affected_population * 2000; // 4 packs per person for 50 days
            cost.innerText = supplies*50+boats*15000+camps*4000;
            campsText.innerText = camps;
            boatsText.innerText = boats;
            suppliesText.innerText = supplies;

            plotFloodZones(map, parseFloat(magnitude));
            generateResourceChart(camps, boats, supplies);
        } catch (error) {
            console.error("Error fetching predictions:", error);
        }
    }

    function plotFloodZones(map, magnitude) {
        map = new google.maps.Map(mapDiv, {
            center: { lat: 19.076, lng: 72.8777 }, // Mumbai
            zoom: 12
        });

        floodZones.forEach(zone => zone.setMap(null));
        floodZones = [];

        const redZone = new google.maps.Circle({
            strokeColor: "#FF0000",
            fillColor: "#FF0000",
            fillOpacity: 0.35,
            map,
            center: { lat: 19.076, lng: 72.8777 },
            radius: magnitude * 1000 // Red zone radius
        });

        const orangeZone = new google.maps.Circle({
            strokeColor: "#FFA500",
            fillColor: "#FFA500",
            fillOpacity: 0.35,
            map,
            center: { lat: 19.076, lng: 72.8777 },
            radius: magnitude * 2000 // Orange zone radius
        });

        const greenZone = new google.maps.Circle({
            strokeColor: "#008000",
            fillColor: "#008000",
            fillOpacity: 0.35,
            map,
            center: { lat: 19.076, lng: 72.8777 },
            radius: magnitude * 3000 // Green zone radius
        });

        floodZones.push(redZone, orangeZone, greenZone);
    }

    function generateResourceChart(camps, boats, supplies) {
        const ctx = document.getElementById("resourcesChart").getContext("2d");

        new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Relief Camps", "Boats", "Food & Water Packs"],
                datasets: [{
                    label: "Resources Needed",
                    data: [camps, boats, supplies],
                    backgroundColor: ["blue", "green", "red"]
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
