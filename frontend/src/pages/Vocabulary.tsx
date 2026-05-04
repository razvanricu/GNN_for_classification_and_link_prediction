import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchVocabulary } from "../api/vocabulary";

export default function Vocabulary() {
  const [search, setSearch] = useState("");
  const { data, isLoading } = useQuery({
    queryKey: ["vocabulary"],
    queryFn: fetchVocabulary,
  });

  const words = data?.words || [];
  const filtered = search
    ? words.filter((w) => w.word.includes(search.toLowerCase()))
    : words;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">TF-IDF Vocabulary</h1>
        <p className="text-sm text-gray-500 mt-1">
          These {data?.total || 750} words are used as features for all models.
          Each product is represented as a {data?.total || 750}-dimensional vector
          of TF-IDF weights. Words with lower IDF are more common across documents.
        </p>
      </div>

      <div className="flex items-center gap-4">
        <input
          type="text"
          placeholder="Filter words..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm w-64 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <span className="text-sm text-gray-500">
          {filtered.length} / {words.length} words shown
        </span>
      </div>

      {isLoading ? (
        <div className="py-8 text-center text-gray-400">Loading vocabulary...</div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="max-h-[600px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b sticky top-0">
                <tr>
                  <th className="text-left px-4 py-2 font-medium text-gray-600">#</th>
                  <th className="text-left px-4 py-2 font-medium text-gray-600">Word</th>
                  <th className="text-right px-4 py-2 font-medium text-gray-600">IDF Score</th>
                  <th className="text-left px-4 py-2 font-medium text-gray-600">Rarity</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {filtered.map((w, i) => {
                  const maxIdf = words[words.length - 1]?.idf || 1;
                  const minIdf = words[0]?.idf || 0;
                  const pct = ((w.idf - minIdf) / (maxIdf - minIdf)) * 100;
                  return (
                    <tr key={w.word} className="hover:bg-gray-50">
                      <td className="px-4 py-1.5 text-gray-400 text-xs">{i + 1}</td>
                      <td className="px-4 py-1.5 font-mono font-medium text-gray-900">
                        {w.word}
                      </td>
                      <td className="px-4 py-1.5 text-right text-gray-600">{w.idf.toFixed(4)}</td>
                      <td className="px-4 py-1.5 w-48">
                        <div className="w-full bg-gray-100 rounded-full h-2">
                          <div
                            className="h-2 rounded-full bg-indigo-500"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
