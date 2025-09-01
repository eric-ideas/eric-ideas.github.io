// Has to be in the head tag, otherwise a flicker effect will occur.

// Always apply dark mode.
let setThemeSetting = () => {
  localStorage.setItem("theme", "dark");
  document.documentElement.setAttribute("data-theme-setting", "dark");
  applyTheme();
};

// Apply dark theme to the website.
let applyTheme = () => {
  let theme = "dark";

  transTheme();
  setHighlight(theme);
  setGiscusTheme(theme);
  setSearchTheme(theme);

  if (typeof mermaid !== "undefined") setMermaidTheme(theme);
  if (typeof Diff2HtmlUI !== "undefined") setDiff2htmlTheme(theme);
  if (typeof echarts !== "undefined") setEchartsTheme(theme);
  if (typeof Plotly !== "undefined") setPlotlyTheme(theme);
  if (typeof vegaEmbed !== "undefined") setVegaLiteTheme(theme);

  document.documentElement.setAttribute("data-theme", theme);

  // Add class to tables.
  let tables = document.getElementsByTagName("table");
  for (let i = 0; i < tables.length; i++) {
    tables[i].classList.add("table-dark");
  }

  // Set Jupyter notebook themes.
  let jupyterNotebooks = document.getElementsByClassName("jupyter-notebook-iframe-container");
  for (let i = 0; i < jupyterNotebooks.length; i++) {
    let bodyElement = jupyterNotebooks[i].getElementsByTagName("iframe")[0].contentWindow.document.body;
    bodyElement.setAttribute("data-jp-theme-light", "false");
    bodyElement.setAttribute("data-jp-theme-name", "JupyterLab Dark");
  }

  // Update medium-zoom background.
  if (typeof medium_zoom !== "undefined") {
    medium_zoom.update({
      background: getComputedStyle(document.documentElement).getPropertyValue("--global-bg-color") + "ee",
    });
  }
};

let setHighlight = (theme) => {
  document.getElementById("highlight_theme_light").media = "none";
  document.getElementById("highlight_theme_dark").media = "";
};

let setGiscusTheme = (theme) => {
  function sendMessage(message) {
    const iframe = document.querySelector("iframe.giscus-frame");
    if (!iframe) return;
    iframe.contentWindow.postMessage({ giscus: message }, "https://giscus.app");
  }
  sendMessage({ setConfig: { theme: "dark" } });
};

let addMermaidZoom = (records, observer) => {
  var svgs = d3.selectAll(".mermaid svg");
  svgs.each(function () {
    var svg = d3.select(this);
    svg.html("<g>" + svg.html() + "</g>");
    var inner = svg.select("g");
    var zoom = d3.zoom().on("zoom", function (event) {
      inner.attr("transform", event.transform);
    });
    svg.call(zoom);
  });
  observer.disconnect();
};

let setMermaidTheme = () => {
  let theme = "dark";
  document.querySelectorAll(".mermaid").forEach((elem) => {
    let svgCode = elem.previousSibling.childNodes[0].innerHTML;
    elem.removeAttribute("data-processed");
    elem.innerHTML = svgCode;
  });
  mermaid.initialize({ theme: theme });
  window.mermaid.init(undefined, document.querySelectorAll(".mermaid"));
  const observable = document.querySelector(".mermaid svg");
  if (observable !== null) {
    var observer = new MutationObserver(addMermaidZoom);
    observer.observe(observable, { childList: true });
  }
};

let setDiff2htmlTheme = () => {
  document.querySelectorAll(".diff2html").forEach((elem) => {
    let textData = elem.previousSibling.childNodes[0].innerHTML;
    elem.innerHTML = "";
    const diff2htmlUi = new Diff2HtmlUI(elem, textData, {
      colorScheme: "dark",
      drawFileList: true,
      highlight: true,
      matching: "lines",
    });
    diff2htmlUi.draw();
  });
};

let setEchartsTheme = () => {
  document.querySelectorAll(".echarts").forEach((elem) => {
    let jsonData = elem.previousSibling.childNodes[0].innerHTML;
    echarts.dispose(elem);
    let chart = echarts.init(elem, "dark-fresh-cut");
    chart.setOption(JSON.parse(jsonData));
  });
};

let setPlotlyTheme = () => {
  document.querySelectorAll(".js-plotly-plot").forEach((elem) => {
    let jsonData = JSON.parse(elem.previousSibling.childNodes[0].innerHTML);
    const plotlyDarkLayout = { layout: { paper_bgcolor: "rgb(17,17,17)", plot_bgcolor: "rgb(17,17,17)" } };
    jsonData.layout = { ...plotlyDarkLayout, ...jsonData.layout };
    Plotly.relayout(elem, jsonData.layout);
  });
};

let setVegaLiteTheme = () => {
  document.querySelectorAll(".vega-lite").forEach((elem) => {
    let jsonData = elem.previousSibling.childNodes[0].innerHTML;
    elem.innerHTML = "";
    vegaEmbed(elem, JSON.parse(jsonData), { theme: "dark" });
  });
};

let setSearchTheme = () => {
  const ninjaKeys = document.querySelector("ninja-keys");
  if (!ninjaKeys) return;
  ninjaKeys.classList.add("dark");
};

let transTheme = () => {
  document.documentElement.classList.add("transition");
  window.setTimeout(() => {
    document.documentElement.classList.remove("transition");
  }, 500);
};

// Init always with dark theme
let initTheme = () => {
  setThemeSetting();
  document.addEventListener("DOMContentLoaded", function () {
    applyTheme();
  });
};
