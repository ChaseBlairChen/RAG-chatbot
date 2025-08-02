// utils/constants.ts
import type { AnalysisTool } from '../types';

export const ANALYSIS_TOOLS: AnalysisTool[] = [
  {
    id: 'comprehensive',
    title: 'Complete Document Analysis',
    description: 'Run all analysis tools at once for comprehensive insights',
    prompt: 'Provide a comprehensive legal analysis including: summary, key clauses, risks, timeline, obligations, and missing clauses.',
    icon: '🔍',
    category: 'Complete',
    idealFor: ['Any legal document'],
    riskLevel: 'low',
    isComprehensive: true
  },
  {
    id: 'summarize',
    title: 'Legal Document Summarization',
    description: 'Get plain English summaries while keeping legal tone intact',
    prompt: 'Summarize this legal document in plain English, keeping the legal tone intact. Highlight purpose, parties involved, and key terms.',
    icon: '📄',
    category: 'Analysis',
    idealFor: ['Contracts', 'Case briefs', 'Discovery documents'],
    riskLevel: 'low'
  },
  {
    id: 'extract-clauses',
    title: 'Key Clause Extraction',
    description: 'Extract termination, indemnification, liability clauses automatically',
    prompt: 'Extract and list the clauses related to termination, indemnification, liability, governing law, and confidentiality.',
    icon: '📋',
    category: 'Extraction',
    idealFor: ['NDAs', 'Employment agreements', 'Service contracts'],
    riskLevel: 'low'
  },
  {
    id: 'missing-clauses',
    title: 'Missing Clause Detection',
    description: 'Flag commonly expected clauses that might be missing',
    prompt: 'Analyze this contract and flag any commonly expected legal clauses that are missing, such as limitation of liability or dispute resolution.',
    icon: '⚠️',
    category: 'Risk Assessment',
    idealFor: ['Startup contracts', 'Vendor agreements'],
    riskLevel: 'medium'
  },
  {
    id: 'risk-flagging',
    title: 'Legal Risk Flagging',
    description: 'Identify clauses that may pose legal risks to signing party',
    prompt: 'Identify any clauses that may pose legal risks to the signing party, such as unilateral termination, broad indemnity, or vague obligations.',
    icon: '🚩',
    category: 'Risk Assessment',
    idealFor: ['Lease agreements', 'IP transfer agreements'],
    riskLevel: 'high'
  },
  {
    id: 'timeline-extraction',
    title: 'Timeline & Deadline Extraction',
    description: 'Extract all dates, deadlines, and renewal periods',
    prompt: 'Extract and list all dates, deadlines, renewal periods, and notice periods mentioned in this document.',
    icon: '📅',
    category: 'Extraction',
    idealFor: ['Leases', 'Licensing deals'],
    riskLevel: 'low'
  },
  {
    id: 'obligations',
    title: 'Obligation Summary',
    description: 'List all required actions and obligations with deadlines',
    prompt: 'List all actions or obligations the signing party is required to perform, along with associated deadlines or conditions.',
    icon: '✅',
    category: 'Analysis',
    idealFor: ['Service contracts', 'Compliance agreements'],
    riskLevel: 'low'
  }
];

export const TEST_ACCOUNTS = [
  { username: 'demo', password: 'demo123', email: 'demo@legalassistant.ai', role: 'user', subscription_tier: 'free' },
  { username: 'tester1', password: 'test123', email: 'tester1@company.com', role: 'user', subscription_tier: 'premium' },
  { username: 'tester2', password: 'test456', email: 'tester2@company.com', role: 'user', subscription_tier: 'free' },
  { username: 'lawyer1', password: 'legal123', email: 'lawyer1@lawfirm.com', role: 'user', subscription_tier: 'premium' }
];


const DEFAULT_BACKEND_URL = "https://day-sons-grown-sides.trycloudflare.com";


