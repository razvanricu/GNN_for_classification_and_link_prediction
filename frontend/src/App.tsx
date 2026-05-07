import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AppLayout from "./components/layout/AppLayout";
import Dashboard from "./pages/Dashboard";
import Catalog from "./pages/Catalog";
import ProductDetail from "./pages/ProductDetail";
import InductiveDemo from "./pages/InductiveDemo";
import Analytics from "./pages/Analytics";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 60_000, retry: 1 } },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="catalog" element={<Catalog />} />
            <Route path="product/:asin" element={<ProductDetail />} />
            <Route path="inductive" element={<InductiveDemo />} />
            <Route path="analytics" element={<Analytics />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
