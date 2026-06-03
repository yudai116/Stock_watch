import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import "./globals.css";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Stock Watch | スイングトレード監視",
  description: "日米株式スイングトレード向け株価監視ダッシュボード",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja" className={`${geistMono.variable}`}>
      <body className="bg-gray-950 text-white antialiased min-h-screen font-mono">
        <nav className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
            <a href="/" className="flex items-center gap-2">
              <span className="text-blue-400 text-xl">📈</span>
              <span className="font-bold text-white">Stock Watch</span>
              <span className="text-xs text-gray-500 ml-1 hidden sm:inline">スイングトレード</span>
            </a>
            <div className="flex items-center gap-4">
              <a href="/backtest" className="text-xs text-gray-400 hover:text-white transition-colors">
                バックテスト結果
              </a>
              <a href="/paper-trading" className="text-xs text-gray-400 hover:text-white transition-colors">
                ペーパートレード
              </a>
              <span className="text-xs text-gray-600">Powered by Yahoo Finance</span>
            </div>
          </div>
        </nav>
        <main className="max-w-2xl mx-auto px-4 py-5">{children}</main>
      </body>
    </html>
  );
}
