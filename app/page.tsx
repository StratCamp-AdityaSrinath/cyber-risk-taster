// This is the corrected TypeScript code for your page.tsx file.
'use client'; 
import { useState } from 'react';

// --- TYPE DEFINITIONS ---
// This tells TypeScript what the structure of our results object will be.
type SimulationResult = {
  pure_premium_mean: number;
  var_95: number;
  var_99: number;
};

// --- MOCK DATA ---
const industries = [
  { name: 'Agriculture (NAICS 11)', naics: 11 },
  { name: 'Construction (NAICS 23)', naics: 23 },
  { name: 'Manufacturing (NAICS 31)', naics: 31 },
  { name: 'Public Administration (NAICS 92)', naics: 92 },
];

const employeeSizes = [
  "<5", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
  "35-39", "40-49", "50-74", "75-99", "100-149", "150-199",
  "200-299", "300-399", "400-499", "500-749", "750-999",
  "1,000-1,499", "1,500-1,999", "2,000-2,499", "2,500-4,999", "5,000+"
];

const cyberEvents = [
    { code: 'RN.01', name: 'Ransomware' },
    { code: 'PHV.01', name: 'Phishing Breach' },
    { code: 'SC.01', name: 'Supply Chain Compromise' },
    { code: 'BD.01', name: 'Biometric Data Theft' },
];

export default function HomePage() {
  // --- STATE VARIABLES with explicit types ---
  const [industry, setIndustry] = useState<number>(11);
  const [employeeSize, setEmployeeSize] = useState<string>('100-149');
  const [deductible, setDeductible] = useState<number>(50000);
  const [selectedEvents, setSelectedEvents] = useState<string[]>(['RN.01']);
  
  // The results can be of type SimulationResult OR null.
  const [results, setResults] = useState<SimulationResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  // --- HANDLERS with explicit types ---
  // The parameter 'eventCode' is explicitly defined as a string.
  const handleEventChange = (eventCode: string) => {
    setSelectedEvents(prev =>
      prev.includes(eventCode)
        ? prev.filter(code => code !== eventCode)
        : [...prev, eventCode]
    );
  };

  const handleRunSimulation = async () => {
    setIsLoading(true);
    setError('');
    setResults(null);
    const simulationInput = {
      industry_naics: industry, // No need for parseInt now
      employee_size: employeeSize,
      deductible,
      selected_events: selectedEvents,
    };
    try {
      const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(simulationInput),
      });
      if (!response.ok) throw new Error('Network response was not ok');
      const data = await response.json();
      if (data.error) setError(data.error);
      else setResults(data);
    } catch (err) {
      setError('Failed to run simulation.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    // Note there is no 'styles.main' here. We will use global CSS.
    <main>
      <div className="container">
        <h1>Cyber Risk Taster Model</h1>
        <div className="controls">
          {/* Industry: e.target.value is a string, so we must convert it to a number */}
          <div className="controlGroup">
            <label htmlFor="industry">Industry (NAICS)</label>
            <select id="industry" value={industry} onChange={(e) => setIndustry(Number(e.target.value))}>
              {industries.map(ind => <option key={ind.naics} value={ind.naics}>{ind.name}</option>)}
            </select>
          </div>
          {/* Employee Size */}
          <div className="controlGroup">
            <label htmlFor="employeeSize">Employee Count</label>
            <select id="employeeSize" value={employeeSize} onChange={(e) => setEmployeeSize(e.target.value)}>
              {employeeSizes.map(size => <option key={size} value={size}>{size}</option>)}
            </select>
          </div>
          {/* Deductible */}
          <div className="controlGroup">
            <label htmlFor="deductible">Deductible: ${deductible.toLocaleString()}</label>
            <input type="range" id="deductible" min="0" max="250000" step="10000" value={deductible} onChange={(e) => setDeductible(Number(e.target.value))} />
          </div>
          {/* Events */}
          <div className="controlGroup">
            <label>Adverse Cyber Events</label>
            <div className="checkboxGroup">
                {cyberEvents.map(event => (
                    <div key={event.code}>
                        <input type="checkbox" id={event.code} value={event.code} checked={selectedEvents.includes(event.code)} onChange={() => handleEventChange(event.code)} />
                        <label htmlFor={event.code}>{event.name}</label>
                    </div>
                ))}
            </div>
          </div>
        </div>
        <button onClick={handleRunSimulation} disabled={isLoading || selectedEvents.length === 0}>
          {isLoading ? 'Simulating...' : 'Run Simulation'}
        </button>
        {error && <p className="error">{error}</p>}
        {/* We can now safely access properties on 'results' because TypeScript knows its shape. */}
        {results && (
          <div className="results">
            <h2>Simulation Results</h2>
            <p><strong>Mean Annual Premium:</strong> ${results.pure_premium_mean.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
            <p><strong>95% Value at Risk (VaR):</strong> ${results.var_95.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
            <p><strong>99% Value at Risk (VaR):</strong> ${results.var_99.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
          </div>
        )}
      </div>
    </main>
  );
}