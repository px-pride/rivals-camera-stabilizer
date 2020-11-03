const path = window.require('path')

const PY_DIST_FOLDER = 'dist'
const PY_FOLDER = 'python'
const PY_MODULE = 'rivals_stabilizer' // without .py suffix

let pyProc = null

const guessPackaged = () => {
  const fullPath = path.join(path.dirname(__dirname), PY_DIST_FOLDER)
  return require('fs').existsSync(fullPath)
}

const getScriptPath = () => {
  if (!guessPackaged()) {
    return path.join(path.dirname(__dirname), PY_FOLDER, PY_MODULE + '.py')
  }
  if (process.platform === 'win32') {
    return path.join(path.dirname(__dirname), PY_DIST_FOLDER, PY_MODULE, PY_MODULE + '.exe')
  }
  return path.join(path.dirname(__dirname), PY_DIST_FOLDER, PY_MODULE, PY_MODULE)
}

let infile = document.getElementById('infile')
let outfile = document.querySelector('#outfile')
let out_xres = document.querySelector('#out_xres')
let out_yres = document.querySelector('#out_yres')
let proc_xres = document.querySelector('#proc_xres')
let proc_yres = document.querySelector('#proc_yres')
let pad_x = document.querySelector('#pad_x')
let pad_y = document.querySelector('#pad_y')
let feature_sparsity = document.querySelector('#feature_sparsity')
let feature_threshold = document.querySelector('#feature_threshold')
let draw_features = document.querySelector('#draw_features')
let button = document.querySelector('#but1')
let result = document.querySelector('#result')
var file = null

function sendToPython(infile, outfile, out_xres, out_yres, proc_xres, proc_yres, pad_x, pad_y, feature_sparsity, feature_threshold, draw_features) {
  let script = getScriptPath()
  console.log(script)
  if (guessPackaged()) {
    pyProc = require('child_process').execFile(script, [infile, outfile, out_xres, out_yres, proc_xres, proc_yres, pad_x, pad_y, feature_sparsity, feature_threshold, draw_features])
  } else {
    pyProc = require('child_process').execFile(script, [infile, outfile, out_xres, out_yres, proc_xres, proc_yres, pad_x, pad_y, feature_sparsity, feature_threshold, draw_features])
  }
 
  if (pyProc != null) {
    //console.log(pyProc)
    console.log('child process success')
  }

}

infile.onchange = e => {
  file = e.target.files[0]
}

button.addEventListener('click', () => {
  sendToPython(
    file.path,
    "./" + outfile.value,
    out_xres.value,
    out_yres.value,
    proc_xres.value,
    proc_yres.value,
    pad_x.value,
    pad_y.value,
    feature_sparsity.value,
    feature_threshold.value,
    draw_features.value);
})

