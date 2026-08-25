const path = require('path');

// Mnemosyne MCP server under PM2.
// Paths resolve relative to THIS file so the config is portable.
// IMPORTANT: PYTHONPATH is blanked on purpose. Hermes Agent exports a
// PYTHONPATH pointing at its own venv; inheriting it makes this process
// import hermes's mcp package (missing pywintypes on Windows) instead of
// the project .venv. Keep env minimal and self-contained.
module.exports = {
  apps: [{
    name: 'mnemosyne',
    script: path.join(__dirname, '.venv', 'Scripts', 'python.exe'),
    args: '-m server.main',
    cwd: __dirname,
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '2G',
    env: {
      PYTHONPATH: '',
      PYTHONUTF8: '1',
      PYTHONIOENCODING: 'utf-8'
    },
    error_file: path.join(__dirname, 'logs', 'pm2-error.log'),
    out_file: path.join(__dirname, 'logs', 'pm2-output.log'),
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
  }]
};
