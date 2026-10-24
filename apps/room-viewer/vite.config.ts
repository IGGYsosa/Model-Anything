import { createReadStream, statSync } from 'node:fs';
import { extname, resolve, sep } from 'node:path';
import { defineConfig, type Plugin } from 'vite';

const app = import.meta.dirname;
const repository = resolve(app, '../..');

/** Dev only: serve local pipeline output at the job server's paths, so
 * ?scene=/jobs/<id>/scene/scene.json works the same against `npm run dev`. */
function localScenes(): Plugin {
  const roots: Record<string, string> = { '/jobs/': resolve(repository, 'data/jobs'), '/scenes/': resolve(repository, 'data/scenes') };
  const types: Record<string, string> = { '.json': 'application/json', '.ply': 'application/octet-stream',
    '.sog': 'application/octet-stream', '.spz': 'application/octet-stream', '.webp': 'image/webp' };
  return {
    name: 'local-scenes',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = decodeURIComponent((req.url ?? '').split('?')[0]);
        const prefix = Object.keys(roots).find(p => path.startsWith(p));
        if (!prefix) return next();
        const file = resolve(roots[prefix], '.' + path.slice(prefix.length - 1));
        if (!file.startsWith(roots[prefix] + sep)) return next();
        try {
          if (!statSync(file).isFile()) return next();
        } catch { return next(); }
        res.setHeader('Content-Type', types[extname(file)] ?? 'application/octet-stream');
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  root: app,
  base: process.env.VIEWER_BASE || '/',
  plugins: [localScenes()],
  optimizeDeps: { include: ['three/addons/objects/Reflector.js'] },
  publicDir: resolve(repository, 'data/web-assets'),
  server: {
    host: '127.0.0.1', port: 5188, strictPort: true, fs: { allow: [repository] },
    // Render evidence is output, not application source. Publishing a preview
    // image must not reload every viewer/capture in the middle of a build.
    watch: { ignored: ['**/data/captures/**', '**/data/recordings/**', '**/data/web-assets/office-v4/live/**', '**/data/jobs/**', '**/data/scenes/**'] },
  },
  build: {
    outDir: resolve(app, 'dist'),
    rolldownOptions: {
      input: {
        main: resolve(app, 'index.html'),
        play: resolve(app, 'play.html'),
        capture: resolve(app, 'tools/capture.html'),
      },
    },
  },
});
