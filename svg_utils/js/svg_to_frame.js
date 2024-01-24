/* uniformly named URL object */
var DOMURL = window.URL || window.webkitURL || window;

/* our snapshotting class */
function svg_snapshot(svg_ref) {

    /* DOM object element */
    this.svg_ref = svg_ref;

    /* svg xml root */
    this.svg_root = svg_ref.contentDocument.documentElement;
    this.svg_doc = svg_ref.contentDocument

    /* frames per second */
    this.fps = parseFloat(this.svg_doc.children[0].getAttribute("data-frame-rate"))
    this.total_frames = parseFloat(this.svg_doc.children[0].getAttribute("data-duration"))

    /* total animation duration in seconds */
    this.seconds = this.total_frames / this.fps
    console.log(this.fps)
    console.log(this.total_frames)
    console.log(this.seconds)


    this.svg_root.setCurrentTime(0)
    this.svg_root.pauseAnimations()

    this.updateTimer = () => {
        const t = `${this.svg_root.getCurrentTime().toFixed(0)}s`;
        document.getElementById("t").textContent = t;
    }

    this.setCurrentTime = (t) => {
        this.svg_root.setCurrentTime(t)
    }

    this.pause = () => {
        this.svg_root.pauseAnimations()
    }

    this.play = () => {
        this.svg_root.unpauseAnimations()
    }

    this.stop = () => {
        clearInterval();
        setCurrentTime(0);
        this.svg_root.pauseAnimations()
        updateTimer();
    }

    // move animation into the next frame
    this.step = (offset) => {
        this.svg_root.pauseAnimations();
        let currentTime = this.svg_root.getCurrentTime()

        let newTime = currentTime + offset / this.fps
        if (newTime > this.seconds) {
            newTime = this.seconds
        }

        if (newTime < 0) {
            newTime = 0
        }

        this.svg_root.setCurrentTime(newTime)
        this.screenshot(this.reconstructSVG())
        update_current_frame_number(this.svg_root.getCurrentTime(), this.fps)
        document.getElementById("svg-slider").value = Math.round(newTime / this.seconds * 100)
    }

    // render current animation frame as SVG string
    this.reconstructSVG = () => {
        let svg_element = this.svg_doc.children[0]
        let clone = svg_element.cloneNode(true)

        let gs = svg_element.getElementsByTagName("g")
        let gs_clone = clone.getElementsByTagName("g")

        for (let i = 0; i < gs.length; i++) {

            // g
            let g_node = gs[i]
            let g_transform = window.getComputedStyle(g_node).transform

            let g_node_clone = gs_clone[i];
            let g_node_clone_animateTransform = g_node_clone.children[1] // hack animateTransform
            g_node_clone.removeChild(g_node_clone_animateTransform)

            if (g_transform != "none") {
                g_node_clone.setAttribute("transform", g_transform)
            }

            // child
            let shape = g_node.children[0]
            let shape_display = window.getComputedStyle(shape).display
            let shape_transform = window.getComputedStyle(shape).transform

            let shape_clone = g_node_clone.children[0]
            // let shape_clone_animate = shape_clone.children[0] // hack animate
            this.removeAllChildNodes(shape_clone)

            shape_clone.setAttribute("display", shape_display)
            if (shape_transform != "none") {
                shape_clone.setAttribute("transform", shape_transform)
            }
        }
        return new XMLSerializer().serializeToString(clone);
    }

    // render static SVG to canvas and then PNG. Not used right now
    this.screenshot = (svgString) => {
        var canvas = document.getElementById("canvas");

        canvas.setAttribute("width", this.svg_ref.clientWidth);
        canvas.setAttribute("height", this.svg_ref.clientHeight);

        var ctx = canvas.getContext("2d");
        var img = new Image();
        var svg = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
        var url = DOMURL.createObjectURL(svg);
        img.onload = function () {
            ctx.drawImage(img, 0, 0);
            var png = canvas.toDataURL("image/png");
            // document.querySelector('#png-container').innerHTML = '<img src="' + png + '" />';
            // DOMURL.revokeObjectURL(png);

            // var finalImg = document.createElement("IMG");

            // finalImg.src = png;
            // // finalImg.style.border = "1px solid black";

            // document.getElementById("image_stack").appendChild(finalImg);
        };
        img.src = url;
    }

    this.removeAllChildNodes = (parent) => {
        while (parent.firstChild) {
            parent.removeChild(parent.firstChild);
        }
    }

    this.padWithZero = (num, targetLength) => {
        return String(num).padStart(targetLength, '0');
    }


    this.make_step = (frame_counter, zip) => {

        this.svg_root.pauseAnimations()
        let currentTime = this.svg_root.getCurrentTime()
        let newTime = currentTime + 1 / this.fps

        if (newTime > this.seconds) {
            // animation ended
            console.log(document.getElementById('image_stack').children)
            zip.generateAsync({ type: "blob" }).then(blob => {
                saveAs(blob, `frames.zip`);
            });
            return false;
        }

        this.svg_root.setCurrentTime(newTime)

        // save current SVG as string
        let svgString = this.reconstructSVG()

        // save as blob
        let svg = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
        
        // create data url (creates browsers interal blob: data link)
        let data_url = DOMURL.createObjectURL(svg);
        
        // create bitmap
        let img = new Image();

        // save class reference
        let self = this;

        // mount load process
        img.onload = function () {
            self.make_step_next(frame_counter, this, zip);
        };

        // set image url
        img.src = data_url;
    };

    this.make_step_next = (frame_counter, img, zip) => {

        /* create canvas */
        var canvas = document.createElement("canvas");

        canvas.setAttribute("width", this.svg_ref.clientWidth);
        canvas.setAttribute("height", this.svg_ref.clientHeight);

        // canvas.style.border = "1px solid black";

        /* get canvas 2d contextr */
        var ctx = canvas.getContext('2d');

        /* drav loaded image onto it */
        ctx.drawImage(img, 0, 0);

        /* here we can get dataURL (base64 encoded url with image content) */
        var dataURL = canvas.toDataURL('image/png');

        /*
            and here you can do whatever you want - send image
            by ajax (that base64 encoded url which you can decode
            on serverside) or draw somewhere on page
        */
        var finalImg = document.createElement("IMG");

        finalImg.src = dataURL;
        // finalImg.style.border = "1px solid black";

        document.getElementById("image_stack").appendChild(finalImg);

        // Create a ZIP file we'll add images to
        zip.file(this.padWithZero(frame_counter, 3) + ".png", dataURL.split(';base64,')[1], { base64: true });
        
        this.make_step(frame_counter + 1, zip)
    };

}

let item_ref = null

let animate_svg = () => {
    let zip = new JSZip();

    item_ref.setCurrentTime(0)
    item_ref.make_step(1, zip);
};


let play_with_audio = (itemRef) => {
    let audio = document.getElementById("audio");
    audio.currentTime = 0;
    audio.play();
    itemRef.setCurrentTime(0)
    itemRef.play();
}

let limit_play_time = (e, time_limit) => {
    // Trying to stop the player if it goes above 1 second
    if (e.currentTime > time_limit) {
        e.pause();
        // e.currentTime = 0
    }
}

let update_current_frame_number = (time, fps) => {
    document.getElementById("frame-number").innerHTML = "Current frame: " + Math.round(time * fps).toString()
}


document.addEventListener("DOMContentLoaded", function () {
    let svg_slider = document.getElementById("svg-slider");
    svg_slider.oninput = (event) => {
        let t = parseFloat(event.currentTarget.value) / 100 * item_ref.seconds
        item_ref.svg_root.pauseAnimations()
        item_ref.setCurrentTime(t)
        update_current_frame_number(t, item_ref.fps)
    }
});
