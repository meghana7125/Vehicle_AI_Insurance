const stage = document.getElementById("vehicleStage");
const vehicle = document.getElementById("vehicle");

if (stage && vehicle) {

    stage.addEventListener("mousemove", function(e) {

        const rect = stage.getBoundingClientRect();

        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const rotateY = (x - centerX) / 35;
        const rotateX = (centerY - y) / 45;

        vehicle.style.transform =
            `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;

    });

    stage.addEventListener("mouseleave", function() {

        vehicle.style.transform =
            "rotateX(0deg) rotateY(0deg)";

    });

}